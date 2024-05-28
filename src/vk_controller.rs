use std::{borrow::Cow, collections::{HashMap, HashSet}, rc::Rc, sync::{Arc, RwLock}};

use ash::{extensions::{ext::DebugUtils, khr::{Surface, Swapchain}}, vk::{self, DebugUtilsMessengerCreateInfoEXT, DescriptorSetLayoutBinding, DeviceCreateInfo, DeviceQueueCreateInfo, ExtDescriptorIndexingFn, Image, ImageView, InstanceCreateInfo, PhysicalDevice, Queue, StructureType, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR}, Device, Entry, Instance};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::Window;

use crate::{graphics_objects::{GraphicsObject, Renderable, ResourceID}, pipeline_manager::{ObjectTypeGraphicsResourceType, PipelineConfig, PipelineManager, Vertex}, sampler_manager::SamplerManager, object_manager::ObjectManager, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable, VkAllocator}};

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub struct ObjectID(pub usize);

#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub struct ReferenceObjectID(pub ObjectID);

type FrameCounter = usize;
#[derive(Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Clone, Copy)]
pub struct VerticesIndicesHash(pub u64);
pub type VertexAllocation = AllocationInfo;
pub type IndexAllocation = AllocationInfo;

#[cfg(debug_assertions)]
const IS_DEBUG_MODE: bool = true;
#[cfg(not(debug_assertions))]
const IS_DEBUG_MODE: bool = false;

pub struct VkController {
    window: Window,
    entry: Entry,
    instance: Rc<Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: PhysicalDevice,
    device: Rc<Device>,
    graphics_queue: Queue,
    present_queue: Queue,
    surface: SurfaceKHR,
    swapchain_loader: Swapchain,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<ImageView>,
    // render_pass: vk::RenderPass,
    // pipeline_layout: vk::PipelineLayout,
    // descriptor_set_layout: vk::DescriptorSetLayout,
    // graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<Vec<vk::CommandBuffer>>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    // vertices: Vec<SimpleVertex>,
    // indices: Vec<u32>,
    // vertex_allocation: Option<AllocationInfo>,
    // index_allocation: Option<AllocationInfo>,
    // object_id_to_pipeline: HashMap<ObjectID, PipelineConfig>,
    // object_id_to_vertices_indices_hash: HashMap<ObjectID, VerticesIndicesHash>,
    // objects_to_render: HashMap<(PipelineConfig, VerticesIndicesHash), ObjectsToRender>,
    // uniform_allocation: Option<AllocationInfo>,
    current_frame: usize,
    pub frame_buffer_resized: bool,
    is_minimized: bool,
    descriptor_pool: vk::DescriptorPool,
    // descriptor_sets: Vec<vk::DescriptorSet>,
    // mip_levels: u32,
    // texture_image_allocation: Option<AllocationInfo>,
    // texture_sampler: vk::Sampler,
    color_image_allocation: Option<AllocationInfo>,
    depth_image_allocation: Option<AllocationInfo>,
    msaa_samples: vk::SampleCountFlags,
    allocator: VkAllocator,
    graphics_pipeline_manager: PipelineManager,
    sampler_manager: SamplerManager,
    object_manager: ObjectManager,
}

#[derive(Debug, Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

// Instance and device management
impl VkController {
    const DEVICE_EXTENSIONS: [*const i8; 2] = [Swapchain::name().as_ptr(), ExtDescriptorIndexingFn::name().as_ptr()];
    pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
    const VALIDATION_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];
    pub const MAX_OBJECT_TYPES:  usize = 1000;

    pub fn new(window: Window, application_name: &str) -> Self {
        let entry = Entry::linked();
        
        let debug_messenger_create_info = if IS_DEBUG_MODE {
            Some(Self::get_debug_messenger_create_info())
        } else {
            None
        };
        let instance = Rc::new(Self::create_instance(&entry, application_name, &window, debug_messenger_create_info.as_ref()));

        let mut debug_messenger = None;
        if IS_DEBUG_MODE {
            debug_messenger = Some(Self::setup_debug_messenger(&entry, &instance, debug_messenger_create_info.unwrap()));
        }

        let surface = Self::create_surface(&entry, &instance, &window);

        let (physical_device, msaa_samples) = Self::pick_physical_device(&entry, &instance, &surface);

        let queue_families = Self::find_queue_families(&entry, &instance, &physical_device, &surface);
        
        let device = Rc::new(Self::create_logical_device(&entry, &instance, &physical_device, &surface));

        let mut allocator = VkAllocator::new(instance.clone(), physical_device, device.clone());

        let (graphics_queue, present_queue) = Self::create_graphics_and_present_queue(&device, &queue_families);

        let swapchain_loader = Swapchain::new(&instance, &device);

        let swapchain = Self::create_swapchain(&entry, &instance, &physical_device,  &surface, &window, &swapchain_loader, &mut allocator);

        let swapchain_images = Self::get_swapchain_images(&swapchain, &swapchain_loader);

        let swapchain_image_format = Self::choose_swap_surface_format(&Self::query_swapchain_support(&entry, &instance, &physical_device, &surface).formats).format;

        let swapchain_extent = Self::choose_swap_extent(&Self::query_swapchain_support(&entry, &instance, &physical_device, &surface).capabilities, &window);
        
        let swapchain_image_views = Self::create_image_views(&device, &swapchain_images, swapchain_image_format, &mut allocator );
        
        let color_image_allocation = Self::create_color_resources(swapchain_image_format, &swapchain_extent, msaa_samples, &mut allocator );
        
        let depth_image_allocation = Self::create_depth_resources(&instance, &physical_device, &swapchain_extent, msaa_samples, &mut allocator );
        
        
        let command_pool = Self::create_command_pool(&device, &queue_families, &mut allocator );

        let descriptor_pool = Self::create_descriptor_pool(&device, &mut allocator );
        let sampler_manager = SamplerManager::new();

        let pipeline_manager = PipelineManager::new(&device, swapchain_image_format, msaa_samples, Self::find_depth_format(&instance, &physical_device), &mut allocator);

        let swapchain_framebuffers = Self::create_framebuffers(&device, &pipeline_manager.get_render_pass().unwrap(), &swapchain_image_views, &swapchain_extent, &depth_image_allocation, &color_image_allocation, &mut allocator );

        // let uniform_allocation = Self::create_uniform_buffers(&mut allocator );

        let mut command_buffers = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            command_buffers.push(Self::create_command_buffers(&device, &command_pool, 1));
        }
        
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = Self::create_sync_objects(&device, &mut allocator );

        Self {
            window,
            entry,
            instance,
            debug_messenger,
            physical_device,
            device,
            graphics_queue,
            present_queue,
            surface,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
            // pipeline_layout,
            // render_pass,
            // descriptor_set_layout,
            // graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            // object_id_to_vertices_indices_hash: HashMap::new(),
            // object_id_to_pipeline: HashMap::new(),
            // objects_to_render,
            // uniform_allocation: Some(uniform_allocation),
            current_frame: 0,
            frame_buffer_resized: false,
            is_minimized: false,
            descriptor_pool,
            // descriptor_sets,
            // texture_image_allocation: Some(texture_image_allocation),
            // texture_sampler,
            color_image_allocation: Some(color_image_allocation),
            depth_image_allocation: Some(depth_image_allocation),
            // mip_levels,
            msaa_samples,
            allocator,
            graphics_pipeline_manager: pipeline_manager,
            sampler_manager,
            object_manager: ObjectManager::new(),
        }
    }

    fn create_instance(entry: &Entry, application_name: &str, window: &Window, debug_create_info: Option<&DebugUtilsMessengerCreateInfoEXT>) -> Instance {
        if IS_DEBUG_MODE && !Self::check_validation_layer_support(entry) {
            panic!("Validation layers requested because of debug mode, but is not available!");
        }

        let app_info = ash::vk::ApplicationInfo {
            s_type: StructureType::APPLICATION_INFO,
            p_application_name: application_name.as_ptr().cast(),
            api_version: ash::vk::make_api_version(0, 1, 3, 0),
            p_engine_name: b"Artewald Engine 2".as_ptr().cast(),
            ..Default::default()
        };
    
        let mut required_instance_extensions = ash_window::enumerate_required_extensions(window.raw_display_handle()).unwrap().to_vec();
        // println!("Adding KhrPortabilityEnumerationFn here might not work!");
        // required_instance_extensions.push(KhrPortabilityEnumerationFn::name().as_ptr());
        if IS_DEBUG_MODE {
            required_instance_extensions.push(DebugUtils::name().as_ptr());
        }

        let mut create_info = InstanceCreateInfo {
            s_type: StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: required_instance_extensions.len() as u32,
            pp_enabled_extension_names: required_instance_extensions.as_ptr(),
            enabled_layer_count: 0,
            ..Default::default()
        };

        // create_info.flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;

        if IS_DEBUG_MODE {
            create_info.enabled_layer_count = Self::VALIDATION_LAYERS.len() as u32;
            create_info.pp_enabled_layer_names = Self::VALIDATION_LAYERS.as_ptr().cast();
            
            create_info.p_next = debug_create_info.unwrap() as *const _ as *const std::ffi::c_void;
        } else {
            create_info.enabled_layer_count = 0;
            create_info.p_next = std::ptr::null();
        }

        unsafe {
            entry.create_instance(&create_info, None)
        }.unwrap()
    }

    fn check_validation_layer_support(entry: &Entry) -> bool {
        let available_layers = entry.enumerate_instance_layer_properties().unwrap();

        let validation_layers = Self::VALIDATION_LAYERS;

        for layer_name in validation_layers {
            let mut layer_found = false;

            for layer_properties in available_layers.iter() {
                let u8_slice: &[u8; 256] = unsafe {std::mem::transmute(&layer_properties.layer_name)};
                let mut current_layer_name = String::new();
                u8_slice.iter().for_each(|byte| {
                    if *byte != 0 {
                        current_layer_name.push(*byte as char);
                    }
                });

                if layer_name == current_layer_name {
                    layer_found = true;
                    break;
                }
            }

            if !layer_found {
                return false;
            }
        }

        true
    }

    fn pick_physical_device(entry: &Entry, instance: &Instance, surface: &SurfaceKHR) -> (PhysicalDevice, vk::SampleCountFlags) {
        let mut device_vec = unsafe {
            instance.enumerate_physical_devices()
        }.expect("Expected to be able to look for physical devices (GPU)!");

        if device_vec.is_empty() {
            panic!("No physical devices found that support Vulkan!");
        }

        device_vec.sort_by_key(|device| Self::rate_physical_device_suitability(instance, device));
        device_vec.reverse();

        let mut chosen_device = None;
        let mut msaa_samples = vk::SampleCountFlags::TYPE_1;

        for device in device_vec.iter() {
            if Self::is_device_suitable(entry, instance, device, surface) {
                msaa_samples = Self::get_max_usable_sample_count(instance, device);
                chosen_device = Some(*device);
                break;
            }
        }

        if let Some(device) = chosen_device {
            (device, msaa_samples)
        } else {
            panic!("No suitable physical device found!");
        }
    }

    fn is_device_suitable(entry: &Entry, instance: &Instance, device: &PhysicalDevice, surface: &SurfaceKHR) -> bool {
        let indices = Self::find_queue_families(entry, instance, device, surface);
        let swapchain_support = Self::query_swapchain_support(entry, instance, device, surface);
        let supported_features = unsafe {
            instance.get_physical_device_features(*device)
        };

        indices.is_complete() && Self::check_device_extension_support(instance, device) && Self::is_swapchain_adequate(&swapchain_support) && supported_features.sampler_anisotropy == vk::TRUE
    }

    fn check_device_extension_support(instance: &Instance, device: &PhysicalDevice) -> bool {
        let available_extensions = unsafe {
            instance.enumerate_device_extension_properties(*device)
        }.unwrap();

        let mut required_extensions = Self::DEVICE_EXTENSIONS.to_vec();

        for extension in available_extensions {
            required_extensions.retain(|required_extension| {
                let u8_slice: &[u8; 256] = unsafe {std::mem::transmute(&extension.extension_name)};
                let mut current_extension_name = String::new();
                u8_slice.iter().for_each(|byte| {
                    if *byte != 0 {
                        current_extension_name.push(*byte as char);
                    }
                });

                current_extension_name != unsafe {std::ffi::CStr::from_ptr(*required_extension)}.to_str().unwrap()
            });
        }

        required_extensions.is_empty()
    }

    fn rate_physical_device_suitability(instance: &Instance, device: &PhysicalDevice) -> i32 {
        let device_properties = unsafe {
            instance.get_physical_device_properties(*device)
        };
        let device_features = unsafe {
            instance.get_physical_device_features(*device)
        };

        let mut score = 0;
        
        if device_properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            score += 1000;
        }

        if device_features.geometry_shader != vk::TRUE {
            return 0;
        }

        score
    }

    fn find_queue_families(entry: &Entry, instance: &Instance, physical_device: &PhysicalDevice, surface: &SurfaceKHR) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices { graphics_family: None, present_family: None };

        let queue_families = unsafe {
            instance.get_physical_device_queue_family_properties(*physical_device)
        };

        for (i, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i as u32);
            }

            let is_present_support = unsafe {
                Surface::new(entry, instance).get_physical_device_surface_support(*physical_device, i as u32, *surface)
            }.unwrap();

            if is_present_support {
                indices.present_family = Some(i as u32);
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_graphics_and_present_queue(device: &Device, indices: &QueueFamilyIndices) -> (Queue, Queue) {
        (unsafe {
            device.get_device_queue(indices.graphics_family.expect("Expected the graphics family to be something, but it's not!"), 0)
        },
        unsafe {
            device.get_device_queue(indices.present_family.expect("Expected the present family to be something, but it's not!"), 0)
        }
        )
    }

    fn create_logical_device(entry: &Entry, instance: &Instance, physical_device: &PhysicalDevice, surface: &SurfaceKHR) -> Device {
        let indices = Self::find_queue_families(entry, instance, physical_device, surface);
        
        let unique_queue_families = HashSet::from([indices.graphics_family.expect("No graphics family index was set!"), indices.present_family.expect("No present family index was set!")]);
        
        let mut queue_create_infos = Vec::new();
        for queue_family in unique_queue_families.iter() {
            let queue_create_info = DeviceQueueCreateInfo {
                s_type: StructureType::DEVICE_QUEUE_CREATE_INFO,
                queue_family_index: *queue_family,
                queue_count: 1,
                p_queue_priorities: [1.0].as_ptr(),
                ..Default::default()
            };

            queue_create_infos.push(queue_create_info);
        }

        let device_features = vk::PhysicalDeviceFeatures {
            sampler_anisotropy: vk::TRUE,
            sample_rate_shading: vk::TRUE, // This may cause performance loss, but it's not required
            fill_mode_non_solid: vk::TRUE, // This is only required for wireframe rendering
            ..Default::default()
        };

        let device_create_info = DeviceCreateInfo {
            s_type: StructureType::DEVICE_CREATE_INFO,
            queue_create_info_count: queue_create_infos.len() as u32,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            p_enabled_features: &device_features,
            pp_enabled_extension_names: Self::DEVICE_EXTENSIONS.as_ptr(),
            enabled_extension_count: Self::DEVICE_EXTENSIONS.len() as u32,
            ..Default::default()
        };

        // This apparently is deprecated, so I'll just leave it out for now
        // if IS_DEBUG_MODE {
        //     let validation_layers = VALIDATION_LAYERS;

        //     device_create_info.enabled_layer_count = validation_layers.len() as u32;
        //     device_create_info.pp_enabled_layer_names = validation_layers.as_ptr().cast();
        // } else {
        //     device_create_info.enabled_layer_count = 0;
        // }
        
        unsafe {
            instance.create_device(*physical_device, &device_create_info, None)
        }.unwrap()
    }

    fn wait_for_device(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }

    pub fn cleanup(&mut self) {
        unsafe {
            self.wait_for_device();

            self.cleanup_swapchain();

            self.sampler_manager.destroy_samplers(&self.device, &mut self.allocator);

            // self.allocator.free_memory_allocation(self.uniform_allocation.take().unwrap()).unwrap();

            self.device.destroy_descriptor_pool(self.descriptor_pool, Some(&self.allocator.get_allocation_callbacks()));

            self.object_manager.destroy_all_objects(&self.device, &self.descriptor_pool, &mut self.allocator);
            
            self.graphics_pipeline_manager.destroy(&self.device, &mut self.allocator);

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.render_finished_semaphores[i], Some(&self.allocator.get_allocation_callbacks()));
                self.device.destroy_semaphore(self.image_available_semaphores[i], Some(&self.allocator.get_allocation_callbacks()));
                self.device.destroy_fence(self.in_flight_fences[i], Some(&self.allocator.get_allocation_callbacks()));
            }

            self.device.destroy_command_pool(self.command_pool, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_all_allocations().unwrap();
            self.device.destroy_device(None);

            if IS_DEBUG_MODE {
                DebugUtils::new(&self.entry, &self.instance).destroy_debug_utils_messenger(self.debug_messenger.unwrap(), None);
            }

            Surface::new(&self.entry, &self.instance).destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// Swapchain management
impl VkController {
    fn create_surface(entry: &Entry, instance: &Instance, window: &Window) -> SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None
            ).unwrap()
        }
    }

    fn query_swapchain_support(entry: &Entry, instance: &Instance, physical_device: &PhysicalDevice, surface: &SurfaceKHR) -> SwapchainSupportDetails {
        unsafe {
            let capabilities = Surface::new(entry, instance).get_physical_device_surface_capabilities(*physical_device, *surface).unwrap();
            let formats = Surface::new(entry, instance).get_physical_device_surface_formats(*physical_device, *surface).unwrap();
            let present_modes = Surface::new(entry, instance).get_physical_device_surface_present_modes(*physical_device, *surface).unwrap();

            SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            }
        }
    }

    fn is_swapchain_adequate(swapchain_support: &SwapchainSupportDetails) -> bool {
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    }

    fn choose_swap_surface_format(available_formats: &Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        println!("The format we are checking for is B8G8R8A8_SRGB!, which might not be what you want!");
        for available_format in available_formats {
            if available_format.format == vk::Format::B8G8R8A8_SRGB && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return *available_format;
            }
        }

        available_formats[0]
    }

    fn choose_swap_present_mode(available_present_modes: &Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
        for available_present_mode in available_present_modes {
            if *available_present_mode == vk::PresentModeKHR::MAILBOX {
                return *available_present_mode;
            }
        }

        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(capabilities: &vk::SurfaceCapabilitiesKHR, window: &Window) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }

        let window_size = window.inner_size();
        vk::Extent2D {
            width: window_size.width.max(capabilities.min_image_extent.width).min(capabilities.max_image_extent.width),
            height: window_size.height.max(capabilities.min_image_extent.height).min(capabilities.max_image_extent.height),
        }
    }

    fn create_swapchain(entry: &Entry, instance: &Instance, physical_device: &PhysicalDevice, surface: &SurfaceKHR, window: &Window, swapchain_loader: &Swapchain, allocator: &mut VkAllocator) -> SwapchainKHR {
        let swapchain_support = Self::query_swapchain_support(entry, instance, physical_device, surface);

        let surface_format = Self::choose_swap_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_swap_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swap_extent(&swapchain_support.capabilities, window);

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count > 0 && image_count > swapchain_support.capabilities.max_image_count {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        let mut swapchain_create_info = SwapchainCreateInfoKHR {
            s_type: StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: *surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            ..Default::default()
        };

        let indices = Self::find_queue_families(entry, instance, physical_device, surface);
        let queue_family_indices = [indices.graphics_family.expect("No graphics family index was set!"), indices.present_family.expect("No present family index was set!")];
        if indices.graphics_family != indices.present_family {
            swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            swapchain_create_info.queue_family_index_count = 2;
            swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
        } else {
            swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
            swapchain_create_info.queue_family_index_count = 0;
            swapchain_create_info.p_queue_family_indices = std::ptr::null();
        }

        unsafe {
            swapchain_loader.create_swapchain(&swapchain_create_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    #[inline(always)]
    fn get_swapchain_images(swapchain: &SwapchainKHR, swapchain_loader: &Swapchain) -> Vec<Image> {
        unsafe {
            swapchain_loader.get_swapchain_images(*swapchain)
        }.unwrap()
    }

    pub fn recreate_swapchain(&mut self) {
        if self.window.inner_size().width == 0 || self.window.inner_size().height == 0 {
            self.is_minimized = true;
            return;
        }
        self.is_minimized = false;

        println!("Recreating swapchain!");

        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        self.cleanup_swapchain();

        self.swapchain = Self::create_swapchain(&self.entry, &self.instance, &self.physical_device, &self.surface, &self.window, &self.swapchain_loader, &mut self.allocator);
        self.swapchain_images = Self::get_swapchain_images(&self.swapchain, &self.swapchain_loader);
        self.swapchain_image_views = Self::create_image_views(&self.device, &self.swapchain_images, self.swapchain_image_format, &mut self.allocator);
        let swapchain_capabilities = Self::query_swapchain_support(&self.entry, &self.instance, &self.physical_device, &self.surface);
        self.swapchain_extent = Self::choose_swap_extent(&swapchain_capabilities.capabilities, &self.window);
        self.color_image_allocation = Some(Self::create_color_resources(self.swapchain_image_format, &self.swapchain_extent, self.msaa_samples, &mut self.allocator));
        self.depth_image_allocation = Some(Self::create_depth_resources(&self.instance, &self.physical_device, &self.swapchain_extent, self.msaa_samples, &mut self.allocator));
        self.swapchain_framebuffers = Self::create_framebuffers(&self.device, &self.graphics_pipeline_manager.get_render_pass().unwrap(), &self.swapchain_image_views, &self.swapchain_extent, self.depth_image_allocation.as_ref().unwrap(), self.color_image_allocation.as_ref().unwrap(), &mut self.allocator);
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.allocator.free_memory_allocation(self.color_image_allocation.take().unwrap()).unwrap();
            self.color_image_allocation = None;
            self.allocator.free_memory_allocation(self.depth_image_allocation.take().unwrap()).unwrap();
            self.depth_image_allocation = None;
            
            self.swapchain_framebuffers.iter().for_each(|framebuffer| {
                self.device.destroy_framebuffer(*framebuffer, Some(&self.allocator.get_allocation_callbacks()));
            });
            self.swapchain_image_views.iter().for_each(|image_view| {
                self.device.destroy_image_view(*image_view, Some(&self.allocator.get_allocation_callbacks()));
            });
            self.swapchain_loader.destroy_swapchain(self.swapchain, Some(&self.allocator.get_allocation_callbacks()));
        }
    }

    fn create_image_views(device: &Device, swapchain_images: &[Image], swapchain_image_format: vk::Format, allocator: &mut VkAllocator) -> Vec<ImageView> {
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());

        for swapchain_image in swapchain_images {
            let view_info = vk::ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                image: *swapchain_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_image_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
    
            let image_view = unsafe {
                device.create_image_view(&view_info, Some(&allocator.get_allocation_callbacks()))
            }.unwrap();
            swapchain_image_views.push(image_view);
        }

        swapchain_image_views
    }
}

// Rendering and graphics pipeline
impl VkController {
    fn get_viewport(swapchain_extent: &vk::Extent2D) -> vk::Viewport {
        vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }

    fn get_scissor(swapchain_extent: &vk::Extent2D) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: 0,
                y: 0,
            },
            extent: *swapchain_extent,
        }
    }

    fn create_framebuffers(device: &Device, render_pass: &vk::RenderPass, swapchain_image_allocations: &[ImageView], swapchain_extent: &vk::Extent2D, depth_image_view: &AllocationInfo, color_image_view: &AllocationInfo, allocator: &mut VkAllocator) -> Vec<vk::Framebuffer> {
        let mut swapchain_framebuffers = Vec::with_capacity(swapchain_image_allocations.len());

        for swapchain_image_view in swapchain_image_allocations.iter() {
            let attachments = [color_image_view.get_image_view().unwrap(), depth_image_view.get_image_view().unwrap(), *swapchain_image_view];

            let framebuffer_create_info = vk::FramebufferCreateInfo {
                s_type: StructureType::FRAMEBUFFER_CREATE_INFO,
                render_pass: *render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
                ..Default::default()
            };

            swapchain_framebuffers.push(unsafe {
                device.create_framebuffer(&framebuffer_create_info, Some(&allocator.get_allocation_callbacks()))
            }.unwrap());
        }

        swapchain_framebuffers
    }

    fn create_command_pool(device: &Device, indices: &QueueFamilyIndices, allocator: &mut VkAllocator) -> vk::CommandPool {

        let pool_info = vk::CommandPoolCreateInfo {
            s_type: StructureType::COMMAND_POOL_CREATE_INFO,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: indices.graphics_family.expect("No graphics family index was set!"),
            ..Default::default()
        };

        unsafe {
            device.create_command_pool(&pool_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_command_buffers(device: &Device, command_pool: &vk::CommandPool, num_buffers: u32) -> Vec<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: num_buffers, //Self::MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe {
            device.allocate_command_buffers(&alloc_info)
        }.unwrap()
    }

    fn record_command_buffer(device: &Device, command_buffer: &vk::CommandBuffer, swapchain_framebuffers: &[vk::Framebuffer], render_pass: &vk::RenderPass, image_index: usize, swapchain_extent: &vk::Extent2D, object_manager: &ObjectManager, pipeline_manager: &mut PipelineManager, current_frame: usize, allocator: &mut VkAllocator) {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_inheritance_info: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            device.begin_command_buffer(*command_buffer, &begin_info)
        }.unwrap();

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }
        ];

        let render_pass_info = vk::RenderPassBeginInfo {
            s_type: StructureType::RENDER_PASS_BEGIN_INFO,
            render_pass: *render_pass,
            framebuffer: swapchain_framebuffers[image_index],
            render_area: vk::Rect2D {
                offset: vk::Offset2D {
                    x: 0,
                    y: 0,
                },
                extent: *swapchain_extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        let viewport = Self::get_viewport(swapchain_extent);
        let scissor = Self::get_scissor(swapchain_extent);

        // let vertex_buffers = [vertex_allocation.get_buffer().unwrap()];
        let offsets = [0_u64];

        unsafe {
            device.cmd_begin_render_pass(*command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            object_manager.borrow_objects_to_render().iter().for_each(|(p_c_k, data_using_p_c)| {
                let mut p_c = p_c_k.clone();
                let pipeline = pipeline_manager.get_or_create_pipeline(&mut p_c, device, swapchain_extent, allocator).unwrap();
                data_using_p_c.object_type_num_instances.iter().for_each(|(object_type, (num_instances, num_indices))| {
                    device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_set_viewport(*command_buffer, 0, &[viewport]);
                    device.cmd_set_scissor(*command_buffer, 0, &[scissor]);
                    device.cmd_bind_vertex_buffers(*command_buffer, 0, &[data_using_p_c.vertices.0.get_buffer().unwrap()], &offsets);
                    device.cmd_bind_index_buffer(*command_buffer, data_using_p_c.indices.0.get_buffer().unwrap(), data_using_p_c.object_type_indices_bytes_indices.get(object_type).unwrap().0.0 as u64, vk::IndexType::UINT32);
                    device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::GRAPHICS, p_c.get_pipeline_layout().unwrap(), 0, &[data_using_p_c.descriptor_sets.get(object_type).unwrap()[current_frame]], &[]);
                    device.cmd_draw_indexed(*command_buffer, num_indices.0 as u32, num_instances.0 as u32, 0, 0, 0);
                });
            });
            device.cmd_end_render_pass(*command_buffer);
            device.end_command_buffer(*command_buffer)
        }.unwrap();
    }

    pub fn draw_frame(&mut self) {

        if self.is_minimized && !self.frame_buffer_resized {
            return;
        }

        unsafe {
            self.device.wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX).unwrap();
        }

        let image_index = match unsafe {
            self.swapchain_loader.acquire_next_image(self.swapchain, u64::MAX, self.image_available_semaphores[self.current_frame], vk::Fence::null())
        } {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.frame_buffer_resized = false;
                self.recreate_swapchain();
                return;
            },
            Err(error) => panic!("Failed to acquire next image: {:?}", error),
        };
        
        unsafe {
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]]).unwrap();
        }

        let cmd_buffer = self.command_buffers[self.current_frame][0];

        self.object_manager.update_objects(&self.device, &self.descriptor_pool, self.current_frame, &mut self.allocator);
        Self::record_command_buffer(&self.device, &cmd_buffer, &self.swapchain_framebuffers, &self.graphics_pipeline_manager.get_render_pass().unwrap(), image_index as usize, &self.swapchain_extent, &self.object_manager, &mut self.graphics_pipeline_manager, self.current_frame, &mut self.allocator);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let submit_info = vk::SubmitInfo {
            s_type: StructureType::SUBMIT_INFO,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: self.command_buffers[self.current_frame].len() as u32,
            p_command_buffers: self.command_buffers[self.current_frame].as_ptr(),
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.queue_submit(self.graphics_queue, &[submit_info], self.in_flight_fences[self.current_frame]).unwrap();
        }


        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: StructureType::PRESENT_INFO_KHR,
            wait_semaphore_count: 1,
            p_wait_semaphores: &self.render_finished_semaphores[self.current_frame],
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr().cast(),
            p_image_indices: &image_index,
            p_results: std::ptr::null_mut(),
            ..Default::default()
        };

        match unsafe {
            self.swapchain_loader.queue_present(self.present_queue, &present_info)
        } {
            Ok(_) => (),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.frame_buffer_resized = false;
                self.recreate_swapchain();
            },
            Err(error) => panic!("Failed to present queue: {:?}", error),
        };
        if self.frame_buffer_resized {
            self.frame_buffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % Self::MAX_FRAMES_IN_FLIGHT;
    }
}

// Synchronization and utilities
impl VkController {
    fn create_sync_objects(device: &Device, allocator: &mut VkAllocator) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut image_available_semaphores = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        let mut in_flight_fences = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            s_type: StructureType::SEMAPHORE_CREATE_INFO,
            ..Default::default()
        };

        let fence_create_info = vk::FenceCreateInfo {
            s_type: StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {

            image_available_semaphores.push(unsafe {
                device.create_semaphore(&semaphore_create_info, Some(&allocator.get_allocation_callbacks()))
            }.unwrap());

            render_finished_semaphores.push(unsafe {
                device.create_semaphore(&semaphore_create_info, Some(&allocator.get_allocation_callbacks()))
            }.unwrap());

            in_flight_fences.push(unsafe {
                device.create_fence(&fence_create_info, Some(&allocator.get_allocation_callbacks()))
            }.unwrap());
        }

        (image_available_semaphores, render_finished_semaphores, in_flight_fences)
    }
}

// Resource management
impl VkController {
    // fn create_uniform_buffers(allocator: &mut VkAllocator) -> AllocationInfo {
    //     let buffer_size = std::mem::size_of::<UniformBufferObject>();

    //     allocator.create_uniform_buffers(buffer_size, Self::MAX_FRAMES_IN_FLIGHT).unwrap()
    // }

    fn create_descriptor_pool(device: &Device, allocator: &mut VkAllocator) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                descriptor_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: Self::MAX_FRAMES_IN_FLIGHT as u32 * Self::MAX_OBJECT_TYPES as u32,
            ..Default::default()
        };

        unsafe {
            device.create_descriptor_pool(&pool_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_depth_resources(instance: &Instance, physical_device: &PhysicalDevice, swapchain_extent: &vk::Extent2D, msaa_samples: vk::SampleCountFlags, allocator: &mut VkAllocator) -> AllocationInfo {
        let depth_format = Self::find_depth_format(instance, physical_device);

        let mut allocation_info = allocator.create_image(swapchain_extent.width, swapchain_extent.height, 1, msaa_samples, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();

        allocator.create_image_view(&mut allocation_info, depth_format, vk::ImageAspectFlags::DEPTH, 1).unwrap();

        allocation_info
    }

    fn find_supported_formats(instance: &Instance, physical_device: &PhysicalDevice, candidates: &[vk::Format], tiling: vk::ImageTiling, features: vk::FormatFeatureFlags) -> Option<vk::Format> {
        for format in candidates {
            let props = unsafe {
                instance.get_physical_device_format_properties(*physical_device, *format)
            };

            if (tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features)) || (tiling == vk::ImageTiling::OPTIMAL && props.optimal_tiling_features.contains(features)) {
                return Some(*format);
            }
        }
        None
    }

    fn find_depth_format(instance: &Instance, physical_device: &PhysicalDevice) -> vk::Format {
        Self::find_supported_formats(instance, physical_device, &[vk::Format::D32_SFLOAT, vk::Format::D32_SFLOAT_S8_UINT, vk::Format::D24_UNORM_S8_UINT], vk::ImageTiling::OPTIMAL, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT).unwrap()
    }

    fn get_max_usable_sample_count(instance: &Instance, physical_device: &PhysicalDevice) -> vk::SampleCountFlags {
        let physical_device_properties = unsafe {
            instance.get_physical_device_properties(*physical_device)
        };

        let count = physical_device_properties.limits.framebuffer_color_sample_counts.min(physical_device_properties.limits.framebuffer_depth_sample_counts);

        if count.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if count.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if count.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if count.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if count.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if count.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
    }

    fn create_color_resources(swapchain_format: vk::Format, swapchain_extent: &vk::Extent2D, num_samples: vk::SampleCountFlags, allocator: &mut VkAllocator) -> AllocationInfo {
        let mut color_allocation = allocator.create_image(swapchain_extent.width, swapchain_extent.height, 1, num_samples, swapchain_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();

        allocator.create_image_view(&mut color_allocation, swapchain_format, vk::ImageAspectFlags::COLOR, 1).unwrap();

        color_allocation
    }
    
    pub fn get_swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    // The object will not be remove until the all frames in flight have passed
    pub fn remove_objects_to_render(&mut self, object_ids: Vec<ObjectID>) {
        self.object_manager.remove_objects(object_ids, &self.device, &self.instance, &self.physical_device, &self.command_pool, &self.descriptor_pool, &self.graphics_queue, &mut self.sampler_manager, self.current_frame, &mut self.allocator);
    }
}

// Debugging and validation
impl VkController {
    fn setup_debug_messenger(entry: &Entry, instance: &Instance, debug_utils_create_info: DebugUtilsMessengerCreateInfoEXT) -> vk::DebugUtilsMessengerEXT {
        let debug_utils_loader = DebugUtils::new(entry, instance);
        match unsafe {
            debug_utils_loader.create_debug_utils_messenger(&debug_utils_create_info, None)
        } {
            Ok(messenger) => messenger,
            Err(e) => panic!("Failed to set up debug messenger: {:?}", e),
        }
    }

    fn get_debug_messenger_create_info() -> DebugUtilsMessengerCreateInfoEXT {
        DebugUtilsMessengerCreateInfoEXT {
            s_type: StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::INFO | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(Self::debug_callback),
            ..Default::default()
        }
    }

    unsafe extern "system" fn debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        _p_user_data: *mut std::ffi::c_void
    ) -> vk::Bool32 {
        
        let debug_type = match message_type {
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "General",
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "Performance",
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "Validation",
            _ => "Unknown",
        };

        let debug_severity = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "Verbose",
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "Info",
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "Warning",
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "Error",
            _ => "Unknown",
        };

        if message_severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
            let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
            println!("[Debug][{debug_type}][{debug_severity}]: {:?}", message);
        }

        vk::FALSE
    }
}


pub trait VkControllerGraphicsObjectsControl<T: Vertex + Clone> {
    // , Vec<(ResourceID, fn() -> ObjectTypeGraphicsResourceType, DescriptorSetLayoutBinding)>)
    fn add_objects_to_render(&mut self, original_objects: Vec<Arc<RwLock<dyn GraphicsObject<T>>>>) -> Result<Vec<(ObjectID, Arc<RwLock<dyn GraphicsObject<T>>>)>, Cow<'static, str>>;
}

impl<T: Vertex + Clone + 'static> VkControllerGraphicsObjectsControl<T> for VkController {
    fn add_objects_to_render(&mut self, original_objects: Vec<Arc<RwLock<dyn GraphicsObject<T>>>>) -> Result<Vec<(ObjectID, Arc<RwLock<dyn GraphicsObject<T>>>)>, Cow<'static, str>> {
        let object_ids = self.object_manager.generate_currently_unused_ids(original_objects.len())?;
        let mut object_id_to_object = Vec::with_capacity(original_objects.len());
        let mut objects_to_render = Vec::with_capacity(original_objects.len());
        let mut i = 0;
        for object in original_objects {
            let object_id = object_ids[i];
            let object_to_render = Box::new(object.clone());
            objects_to_render.push((object_id, object_to_render as Box<dyn Renderable>));
            object_id_to_object.push((object_id, object.clone()));
            i += 1;
        }
        self.object_manager.add_objects(objects_to_render, &self.device, &self.instance, &self.physical_device, &self.command_pool, &self.descriptor_pool, &self.graphics_queue, &mut self.sampler_manager, self.msaa_samples, self.swapchain_image_format, Self::find_depth_format(&self.instance, &self.physical_device), self.current_frame, &mut self.allocator)?;
        Ok(object_id_to_object)
    }
}

struct ObjectsToRender {
    pub vertex_allocation: Option<VertexAllocation>,
    pub index_allocation: Option<IndexAllocation>,
    pub num_indices: u32,
    pub objects: Vec<(ObjectID, Box<dyn Renderable>)>,
}
