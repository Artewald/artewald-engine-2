use std::{borrow::Cow, collections::{HashSet, HashMap, hash_map}, fs::read_to_string, rc::Rc, sync::{Arc, Mutex}, time::Instant};

use ash::{Entry, Instance, vk::{self, DebugUtilsMessengerCreateInfoEXT, DependencyFlags, DeviceCreateInfo, DeviceQueueCreateInfo, Image, ImageSubresourceRange, ImageView, ImageViewCreateInfo, InstanceCreateFlags, InstanceCreateInfo, KhrPortabilityEnumerationFn, PhysicalDevice, Queue, StructureType, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR}, Device, extensions::{khr::{Swapchain, Surface}, ext::DebugUtils}};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use shaderc::{Compiler, ShaderKind};
use winit::window::Window;
use nalgebra_glm as glm;

use crate::{vertex::Vertex, graphics_objects::UniformBufferObject, vk_allocator::{AllocationInfo, VkAllocator}};



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
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // vertex_buffer: vk::Buffer,
    // vertex_buffer_memory: vk::DeviceMemory,
    vertex_allocation: Option<AllocationInfo>,
    // index_buffer: vk::Buffer,
    // index_buffer_memory: vk::DeviceMemory,
    index_allocation: Option<AllocationInfo>,
    // uniform_buffers: Vec<vk::Buffer>,
    // uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_allocations: Vec<AllocationInfo>,
    uniform_buffers_mapping: Vec<*mut std::ffi::c_void>,
    current_frame: usize,
    pub frame_buffer_resized: bool,
    is_minimized: bool,
    start_time: Instant,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    mip_levels: u32,
    // texture_image: vk::Image,
    // texture_image_memory: vk::DeviceMemory,
    // texture_image_view: vk::ImageView,
    texture_image_allocation: Option<AllocationInfo>,
    texture_sampler: vk::Sampler,
    // color_image: vk::Image,
    // color_image_memory: vk::DeviceMemory,
    // color_image_view: vk::ImageView,
    color_image_allocation: Option<AllocationInfo>,
    // depth_image: vk::Image,
    // depth_image_memory: vk::DeviceMemory,
    // depth_image_view: vk::ImageView,
    depth_image_allocation: Option<AllocationInfo>,
    msaa_samples: vk::SampleCountFlags,
    allocator: VkAllocator,
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
    const DEVICE_EXTENSIONS: [*const i8; 1] = [Swapchain::name().as_ptr()];
    const MAX_FRAMES_IN_FLIGHT: usize = 2;
    const VALIDATION_LAYERS: [&'static str; 1] = ["VK_LAYER_KHRONOS_validation"];

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

        let render_pass = Self::create_render_pass(swapchain_image_format, &device, &instance, &physical_device, msaa_samples, &mut allocator );
        
        let descriptor_set_layout = Self::create_descriptor_set_layout(&device, &mut allocator );
        
        let pipeline_layout = Self::create_pipeline_layout(&device, &descriptor_set_layout, &mut allocator );

        let graphics_pipeline = Self::create_graphics_pipeline(&device, &swapchain_extent, &pipeline_layout, &render_pass, msaa_samples, &mut allocator );
        
        let command_pool = Self::create_command_pool(&device, &queue_families, &mut allocator );

        let color_image_allocation = Self::create_color_resources(swapchain_image_format, &swapchain_extent, msaa_samples, &mut allocator );
        
        let depth_image_allocation = Self::create_depth_resources(&instance, &physical_device, &swapchain_extent, msaa_samples, &mut allocator );
        
        let swapchain_framebuffers = Self::create_framebuffers(&device, &render_pass, &swapchain_image_views, &swapchain_extent, &depth_image_allocation, &color_image_allocation, &mut allocator );
        
        let (mut texture_image_allocation, mip_levels) = Self::create_texture_image(&command_pool, &graphics_queue, &mut allocator );

        Self::create_texture_image_view(&mut texture_image_allocation, mip_levels, &mut allocator );

        let texture_sampler = Self::create_texture_sampler(&device, &instance, &physical_device, mip_levels, &mut allocator );

        let (vertices, indices) = Self::load_model("./assets/objects/viking_room.obj");

        let vertex_allocation = Self::create_vertex_buffer(&command_pool, &graphics_queue, &vertices, &mut allocator );

        let index_allocation = Self::create_index_buffer(&command_pool, &graphics_queue, &indices, &mut allocator );

        let (uniform_allocations, uniform_buffers_mapping) = Self::create_uniform_buffers(&device, &mut allocator );
        
        let descriptor_pool = Self::create_descriptor_pool(&device, &mut allocator );
        
        let descriptor_sets = Self::create_descriptor_sets(&device, &descriptor_pool, &uniform_allocations, &descriptor_set_layout, &texture_image_allocation, &texture_sampler);
        
        let command_buffers = Self::create_command_buffers(&device, &command_pool);
        
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
            pipeline_layout,
            render_pass,
            descriptor_set_layout,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            // vertex_buffer,
            // vertex_buffer_memory,
            vertex_allocation: Some(vertex_allocation),
            // index_buffer,
            // index_buffer_memory,
            index_allocation: Some(index_allocation),
            // uniform_buffers,
            // uniform_buffers_memory,
            uniform_allocations,
            uniform_buffers_mapping,
            current_frame: 0,
            frame_buffer_resized: false,
            is_minimized: false,
            start_time: Instant::now(),
            descriptor_pool,
            descriptor_sets,
            // texture_image,
            // texture_image_memory,
            // texture_image_view,
            texture_image_allocation: Some(texture_image_allocation),
            texture_sampler,
            // color_image,
            // color_image_memory,
            // color_image_view,
            color_image_allocation: Some(color_image_allocation),
            // depth_image,
            // depth_image_memory,
            // depth_image_view,
            depth_image_allocation: Some(depth_image_allocation),
            vertices,
            indices,
            mip_levels,
            msaa_samples,
            allocator,
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

        // let surface_loader = Surface::new(entry, instance);

        // let mut supported_present_modes = false;
        // if let Some(index) = indices.graphics_family {
        //     supported_present_modes = unsafe {
        //         surface_loader.get_physical_device_surface_support(*physical_device, index, *surface)
        //     }.unwrap();
        // }
        
        // if supported_present_modes {
        //     indices.present_family = indices.graphics_family;
        // }

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

            self.device.destroy_sampler(self.texture_sampler, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_memory_allocation(self.texture_image_allocation.unwrap()).unwrap();
            self.texture_image_allocation = None;
            //self.device.destroy_image_view(self.texture_image_view, Some(&self.allocator.get_allocation_callbacks()));

            // self.device.destroy_image(self.texture_image, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.free_memory(self.texture_image_memory, Some(&self.allocator.get_allocation_callbacks()));

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                //self.device.destroy_buffer(self.uniform_buffers[i], Some(&self.allocator.get_allocation_callbacks()));
                self.allocator.free_memory_allocation(self.uniform_allocations[i]).unwrap();
                //self.device.free_memory(self.uniform_buffers_memory[i], Some(&self.allocator.get_allocation_callbacks()));
            }
            self.uniform_allocations.clear();

            self.device.destroy_descriptor_pool(self.descriptor_pool, Some(&self.allocator.get_allocation_callbacks()));

            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, Some(&self.allocator.get_allocation_callbacks()));

            self.device.destroy_pipeline(self.graphics_pipeline, Some(&self.allocator.get_allocation_callbacks()));
            self.device.destroy_pipeline_layout(self.pipeline_layout, Some(&self.allocator.get_allocation_callbacks()));
            self.device.destroy_render_pass(self.render_pass, Some(&self.allocator.get_allocation_callbacks()));
            
            // self.device.destroy_buffer(self.index_buffer, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.free_memory(self.index_buffer_memory, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_memory_allocation(self.index_allocation.unwrap()).unwrap();
            self.index_allocation = None;

            // self.device.destroy_buffer(self.vertex_buffer, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.free_memory(self.vertex_buffer_memory, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_memory_allocation(self.vertex_allocation.unwrap()).unwrap();
            self.vertex_allocation = None;

            for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
                self.device.destroy_semaphore(self.render_finished_semaphores[i], Some(&self.allocator.get_allocation_callbacks()));
                self.device.destroy_semaphore(self.image_available_semaphores[i], Some(&self.allocator.get_allocation_callbacks()));
                self.device.destroy_fence(self.in_flight_fences[i], Some(&self.allocator.get_allocation_callbacks()));
            }

            self.device.destroy_command_pool(self.command_pool, Some(&self.allocator.get_allocation_callbacks()));
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
        self.swapchain_framebuffers = Self::create_framebuffers(&self.device, &self.render_pass, &self.swapchain_image_views, &self.swapchain_extent, &self.depth_image_allocation.unwrap(), &self.color_image_allocation.unwrap(), &mut self.allocator);
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            // self.device.destroy_image_view(self.color_image_view, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.destroy_image(self.color_image, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.free_memory(self.color_image_memory, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_memory_allocation(self.color_image_allocation.unwrap()).unwrap();
            self.color_image_allocation = None;
            // self.device.destroy_image_view(self.depth_image_view, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.destroy_image(self.depth_image, Some(&self.allocator.get_allocation_callbacks()));
            // self.device.free_memory(self.depth_image_memory, Some(&self.allocator.get_allocation_callbacks()));
            self.allocator.free_memory_allocation(self.depth_image_allocation.unwrap()).unwrap();
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

    fn create_image_views(device: &Device, swapchain_images: &Vec<Image>, swapchain_image_format: vk::Format, allocator: &mut VkAllocator) -> Vec<ImageView> {
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());

        for i in 0..swapchain_images.len() {
            let view_info = vk::ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                image: swapchain_images[i],
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
            swapchain_image_views.push(image_view);//Self::create_image_view(device, &swapchain_images[i], swapchain_image_format, vk::ImageAspectFlags::COLOR, 1, allocator));
        }

        swapchain_image_views
    }
}

// Rendering and graphics pipeline
impl VkController {
    fn create_graphics_pipeline(device: &Device, swapchain_extent: &vk::Extent2D, pipeline_layout: &vk::PipelineLayout, render_pass: &vk::RenderPass, msaa_samples: vk::SampleCountFlags, allocator: &mut VkAllocator) -> vk::Pipeline {
        let vert_shader_code = Self::compile_shader("./assets/shaders/triangle.vert", ShaderKind::Vertex, "triangle.vert");
        let frag_shader_code = Self::compile_shader("./assets/shaders/triangle.frag", ShaderKind::Fragment, "triangle.frag");

        let vert_shader_module = Self::create_shader_module(device, vert_shader_code, allocator);
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code, allocator);

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: "main".as_ptr().cast(),
            ..Default::default()
        };

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: "main".as_ptr().cast(),
            ..Default::default()
        };


        let binding_description = Vertex::vertex_input_binding_descriptions();
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &binding_description,
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let viewport = Self::get_viewport(swapchain_extent);
        let scissor = Self::get_scissor(swapchain_extent);

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            s_type: StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::TRUE, // This may cause performance loss, but it's not required
            rasterization_samples: msaa_samples,
            min_sample_shading: 0.2,
            p_sample_mask: std::ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A,
            blend_enable: vk::TRUE,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            s_type: StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            s_type: StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            stencil_test_enable: vk::FALSE,
            front: vk::StencilOpState::default(),
            back: vk::StencilOpState::default(),
            ..Default::default()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: *pipeline_layout,
            render_pass: *render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], Some(&allocator.get_allocation_callbacks()))
        }.unwrap()[0];

        // This should always happen at the end
        unsafe {
            device.destroy_shader_module(vert_shader_module, Some(&allocator.get_allocation_callbacks()));
            device.destroy_shader_module(frag_shader_module, Some(&allocator.get_allocation_callbacks()));
        }

        graphics_pipeline
    }

    fn compile_shader(path: &str, shader_kind: ShaderKind, identifier: &str) -> Vec<u32> {
        let compiler = Compiler::new().unwrap();
        let artifact = compiler.compile_into_spirv(&read_to_string(path).unwrap(), shader_kind, identifier, "main", None).unwrap();
        artifact.as_binary().to_owned()
    }

    fn create_shader_module(device: &Device, code: Vec<u32>, allocator: &mut VkAllocator) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: code.len() * std::mem::size_of::<u32>(),
            p_code: code.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device.create_shader_module(&create_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_pipeline_layout(device: &Device, desciptor_set_layout: &vk::DescriptorSetLayout, allocator: &mut VkAllocator) -> vk::PipelineLayout {
        let descriptor_set_layouts = [*desciptor_set_layout];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            device.create_pipeline_layout(&pipeline_layout_create_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_render_pass(swapchain_image_format: vk::Format, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, msaa_samples: vk::SampleCountFlags, allocator: &mut VkAllocator) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_image_format,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let depth_attachment = vk::AttachmentDescription {
            format: Self::find_depth_format(instance, physical_device),
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let color_attachment_resolve = vk::AttachmentDescription {
            format: swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_resolve_ref = vk::AttachmentReference {
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            p_resolve_attachments: &color_attachment_resolve_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let attachments = [color_attachment, depth_attachment, color_attachment_resolve];
        let render_pass_info = vk::RenderPassCreateInfo {
            s_type: StructureType::RENDER_PASS_CREATE_INFO,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,
            ..Default::default()
        };

        unsafe {
            device.create_render_pass(&render_pass_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

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

    fn create_framebuffers(device: &Device, render_pass: &vk::RenderPass, swapchain_image_allocations: &Vec<ImageView>, swapchain_extent: &vk::Extent2D, depth_image_view: &AllocationInfo, color_image_view: &AllocationInfo, allocator: &mut VkAllocator) -> Vec<vk::Framebuffer> {
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

    fn create_command_buffers(device: &Device, command_pool: &vk::CommandPool) -> Vec<vk::CommandBuffer> {

        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool: *command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe {
            device.allocate_command_buffers(&alloc_info)
        }.unwrap()
    }

    fn record_command_buffer(device: &Device, command_buffer: &vk::CommandBuffer, swapchain_framebuffers: &[vk::Framebuffer], render_pass: &vk::RenderPass, image_index: usize, swapchain_extent: &vk::Extent2D, graphics_pipeline: &vk::Pipeline, vertex_allocation: &AllocationInfo, index_allocation: &AllocationInfo, pipeline_layout: &vk::PipelineLayout, descriptor_sets: &Vec<vk::DescriptorSet>, current_frame: usize, indices: &[u32]) {
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

        let vertex_buffers = [vertex_allocation.get_buffer().unwrap()];
        let offsets = [0_u64];

        unsafe {
            device.cmd_begin_render_pass(*command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, *graphics_pipeline);
            device.cmd_set_viewport(*command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(*command_buffer, 0, &[scissor]);
            device.cmd_bind_vertex_buffers(*command_buffer, 0, &vertex_buffers, &offsets);
            device.cmd_bind_index_buffer(*command_buffer, index_allocation.get_buffer().unwrap(), 0, vk::IndexType::UINT32);
            device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &vec![descriptor_sets[current_frame]], &vec![]);
            device.cmd_draw_indexed(*command_buffer, indices.len() as u32, 1, 0, 0, 0);
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

        self.update_uniform_buffer(self.current_frame);

        unsafe {
            self.device.reset_fences(&[self.in_flight_fences[self.current_frame]]).unwrap();
        }

        unsafe {
            self.device.reset_command_buffer(self.command_buffers[self.current_frame], vk::CommandBufferResetFlags::empty()).unwrap();
        }

        Self::record_command_buffer(&self.device, &self.command_buffers[self.current_frame], &self.swapchain_framebuffers, &self.render_pass, image_index as usize, &self.swapchain_extent, &self.graphics_pipeline, &self.vertex_allocation.unwrap(), &self.index_allocation.unwrap(), &self.pipeline_layout, &self.descriptor_sets, self.current_frame, &self.indices);

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let command_buffers = [self.command_buffers[self.current_frame]];

        let submit_info = vk::SubmitInfo {
            s_type: StructureType::SUBMIT_INFO,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: command_buffers.as_ptr(),
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

    fn begin_single_time_command(device: &Device, command_pool: &vk::CommandPool) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: *command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer = unsafe {
            device.allocate_command_buffers(&alloc_info).unwrap()[0]
        };

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            device.begin_command_buffer(command_buffer, &begin_info).unwrap();
        }

        command_buffer
    }

    fn end_single_time_command(device: &Device, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.end_command_buffer(command_buffer).unwrap();
        }

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            device.queue_submit(*graphics_queue, &[submit_info], vk::Fence::null()).unwrap();
            device.queue_wait_idle(*graphics_queue).unwrap();
            device.free_command_buffers(*command_pool, &[command_buffer]);
        }
    }

    fn copy_buffer(src_buffer: &vk::Buffer, dst_buffer: &vk::Buffer, size: vk::DeviceSize, device: &Device, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue) {
        let command_buffer = Self::begin_single_time_command(device, command_pool);

        let copy_region = vk::BufferCopy {
            size,
            ..Default::default()
        };

        unsafe {
            device.cmd_copy_buffer(command_buffer, *src_buffer, *dst_buffer, &[copy_region]);
        }

        Self::end_single_time_command(device, command_pool, graphics_queue, command_buffer);
    }

    fn copy_buffer_to_image(src_buffer: &vk::Buffer, dst_image: &vk::Image, width: u32, height: u32, device: &Device, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue) {
        let command_buffer = Self::begin_single_time_command(device, command_pool);

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: 0,
                y: 0,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        unsafe {
            device.cmd_copy_buffer_to_image(command_buffer, *src_buffer, *dst_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        }

        Self::end_single_time_command(device, command_pool, graphics_queue, command_buffer);
    }
}

// Resource management
impl VkController {
    fn create_vertex_buffer(command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, vertices: &[Vertex], allocator: &mut VkAllocator) -> AllocationInfo {
        allocator.create_device_local_buffer(command_pool, graphics_queue, &bincode::serialize(&vertices).unwrap(), vk::BufferUsageFlags::VERTEX_BUFFER).unwrap()
        
        // let size = std::mem::size_of_val(vertices);
        
        // let (staging_buffer, staging_buffer_memory) = Self::create_buffer(instance, physical_device, device, size as u64, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, allocator);
        
        // unsafe {
        //     let data_ptr = device.map_memory(staging_buffer_memory, 0, size as u64, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
        //     let vertices_ptr = vertices.as_ptr() as *const u8;
        //     std::ptr::copy_nonoverlapping(vertices_ptr, data_ptr, size);
        //     device.unmap_memory(staging_buffer_memory);
        // }
        
        // let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(instance, physical_device, device, size as u64, vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, allocator);

        // Self::copy_buffer(&staging_buffer, &vertex_buffer, size as u64, device, command_pool, graphics_queue);

        // unsafe {
        //     device.destroy_buffer(staging_buffer, Some(&allocator.get_allocation_callbacks()));
        //     device.free_memory(staging_buffer_memory, Some(&allocator.get_allocation_callbacks()));
        // }

        // (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, indices: &[u32], allocator: &mut VkAllocator) -> AllocationInfo {
        allocator.create_device_local_buffer(command_pool, graphics_queue, &bincode::serialize(&indices).unwrap(), vk::BufferUsageFlags::INDEX_BUFFER).unwrap()
        // let size = std::mem::size_of_val(indices);
        
        // let (staging_buffer, staging_buffer_memory) = Self::create_buffer(instance, physical_device, device, size as u64, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, allocator);
        
        // unsafe {
        //     let data_ptr = device.map_memory(staging_buffer_memory, 0, size as u64, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
        //     let indices_ptr = indices.as_ptr() as *const u8;
        //     std::ptr::copy_nonoverlapping(indices_ptr, data_ptr, size);
        //     device.unmap_memory(staging_buffer_memory);
        // }
        
        // let (index_buffer, index_buffer_memory) = Self::create_buffer(instance, physical_device, device, size as u64, vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, allocator);

        // Self::copy_buffer(&staging_buffer, &index_buffer, size as u64, device, command_pool, graphics_queue);

        // unsafe {
        //     device.destroy_buffer(staging_buffer, Some(&allocator.get_allocation_callbacks()));
        //     device.free_memory(staging_buffer_memory, Some(&allocator.get_allocation_callbacks()));
        // }

        // (index_buffer, index_buffer_memory)
    }

    // fn create_buffer(instance: &Instance, physical_device: &PhysicalDevice, device: &Device, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags, allocator: &mut VkAllocator) -> (vk::Buffer, vk::DeviceMemory) {
    //     let buffer_info = vk::BufferCreateInfo {
    //         s_type: StructureType::BUFFER_CREATE_INFO,
    //         size,
    //         usage,
    //         sharing_mode: vk::SharingMode::EXCLUSIVE,
    //         ..Default::default()
    //     };

    //     let buffer = unsafe {
    //         device.create_buffer(&buffer_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap();
        
    //     let memory_requirements = unsafe {
    //         device.get_buffer_memory_requirements(buffer)
    //     };

    //     let alloc_info = vk::MemoryAllocateInfo {
    //         s_type: StructureType::MEMORY_ALLOCATE_INFO,
    //         allocation_size: memory_requirements.size,
    //         memory_type_index: Self::find_memory_type(instance, physical_device, memory_requirements.memory_type_bits, properties).unwrap(),
    //         ..Default::default()
    //     };

    //     println!("Should not allocate memory for many different buffers, but rather have one big buffer and an allocator that uses offsets when binding buffer to memory!");
    //     let buffer_memory = unsafe {
    //         device.allocate_memory(&alloc_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap();

    //     unsafe {
    //         device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
    //     }

    //     (buffer, buffer_memory)
    // }


    fn create_uniform_buffers(device: &Device, allocator: &mut VkAllocator) -> (Vec<AllocationInfo>, Vec<*mut std::ffi::c_void>) {
        let buffer_size = std::mem::size_of::<UniformBufferObject>();

        let mut uniform_buffers = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        // let mut uniform_buffers_memory = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_mapped = Vec::with_capacity(Self::MAX_FRAMES_IN_FLIGHT);
        
        for _ in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let allocation_info = allocator.create_buffer(buffer_size as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap(); //Self::create_buffer(instance, physical_device, device, buffer_size as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, allocator);
            uniform_buffers.push(allocation_info);//uniform_buffer);
            // uniform_buffers_memory.push(uniform_buffer_memory);

            let data_ptr = unsafe {
                device.map_memory(allocation_info.get_memory(), allocation_info.get_memory_start(), buffer_size as u64, vk::MemoryMapFlags::empty()).unwrap()
            };
            uniform_buffers_mapped.push(data_ptr);
        }

        (uniform_buffers, uniform_buffers_mapped)
    }

    fn update_uniform_buffer(&mut self, current_image: usize) {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let mut ubo = UniformBufferObject {
            model: glm::rotate(&glm::identity(), elapsed * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0)),
            view: glm::look_at(&glm::vec3(2.0, 2.0, 2.0), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 0.0, 1.0)),
            proj: glm::perspective(self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32, 90.0_f32.to_radians(), 0.1, 10.0),
        };
        ubo.proj[(1, 1)] *= -1.0;

        unsafe {
            std::ptr::copy_nonoverlapping(&ubo as *const UniformBufferObject as *const std::ffi::c_void, self.uniform_buffers_mapping[current_image], std::mem::size_of::<UniformBufferObject>());
        }
    }

    fn create_descriptor_set_layout(device: &Device, allocator: &mut VkAllocator) -> vk::DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        };

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        };

        let layout_bindings = [ubo_layout_binding, sampler_layout_binding];

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: layout_bindings.len() as u32,
            p_bindings: layout_bindings.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device.create_descriptor_set_layout(&layout_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_descriptor_pool(device: &Device, allocator: &mut VkAllocator) -> vk::DescriptorPool {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            }
        ];

        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets: Self::MAX_FRAMES_IN_FLIGHT as u32,
            ..Default::default()
        };

        unsafe {
            device.create_descriptor_pool(&pool_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }

    fn create_descriptor_sets(device: &Device, descriptor_pool: &vk::DescriptorPool, uniform_buffers: &[AllocationInfo], descriptor_set_layout: &vk::DescriptorSetLayout, texture_allocation: &AllocationInfo, texture_sampler: &vk::Sampler) -> Vec<vk::DescriptorSet> {
        let layouts = [*descriptor_set_layout; Self::MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: *descriptor_pool,
            descriptor_set_count: Self::MAX_FRAMES_IN_FLIGHT as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info).unwrap()
        };

        for i in 0..Self::MAX_FRAMES_IN_FLIGHT {
            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffers[i].get_buffer().unwrap(),
                offset: 0,
                range: std::mem::size_of::<UniformBufferObject>() as u64,
            };

            let image_info = vk::DescriptorImageInfo {
                sampler: *texture_sampler,
                image_view: texture_allocation.get_image_view().unwrap(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let descriptor_writes = [
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: descriptor_sets[i],
                    dst_binding: 0,
                    dst_array_element: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    p_buffer_info: &buffer_info,
                    p_image_info: std::ptr::null(),
                    p_texel_buffer_view: std::ptr::null(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                    dst_set: descriptor_sets[i],
                    dst_binding: 1,
                    dst_array_element: 0,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    p_image_info: &image_info,
                    p_texel_buffer_view: std::ptr::null(),
                    ..Default::default()
                }
            ];

            unsafe {
                device.update_descriptor_sets(&descriptor_writes, &vec![]);
            }
        }

        descriptor_sets
    }

    fn create_texture_image(command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, allocator: &mut VkAllocator) -> (AllocationInfo, u32) {
        let binding = image::open("./assets/images/viking_room.png").unwrap();

        allocator.create_device_local_image(binding, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1).unwrap()
        // let image = binding.to_rgba8();
        // let image_size: vk::DeviceSize = image.dimensions().0 as vk::DeviceSize * image.dimensions().1 as vk::DeviceSize * 4 as vk::DeviceSize;
        
        // let mip_levels = ((image.dimensions().0 as f32).max(image.dimensions().1 as f32).log2().floor() + 1.0) as u32;

        // let (staging_buffer, staging_buffer_memory) = Self::create_buffer(instance, physical_device, device, image_size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, allocator);

        // unsafe {
        //     let data_ptr = device.map_memory(staging_buffer_memory, 0, image_size, vk::MemoryMapFlags::empty()).unwrap() as *mut u8;
        //     std::ptr::copy_nonoverlapping(image.as_ptr(), data_ptr, image_size as usize);
        //     device.unmap_memory(staging_buffer_memory);
        // };

        // let (vk_image, device_memory) = Self::create_image(instance, physical_device, device, image.dimensions().0, image.dimensions().1, mip_levels, vk::SampleCountFlags::TYPE_1, vk::Format::R8G8B8A8_SRGB, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, vk::MemoryPropertyFlags::DEVICE_LOCAL, allocator);

        // Self::transition_image_layout(device, command_pool, graphics_queue, &vk_image, vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mip_levels);
        // Self::copy_buffer_to_image(&staging_buffer, &vk_image, image.dimensions().0, image.dimensions().1, device, command_pool, graphics_queue);
        // //Self::transition_image_layout(device, command_pool, graphics_queue, &vk_image, vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, mip_levels);
        
        // unsafe {
        //     device.destroy_buffer(staging_buffer, Some(&allocator.get_allocation_callbacks()));
        //     device.free_memory(staging_buffer_memory, Some(&allocator.get_allocation_callbacks()));
        // }
        
        // Self::generate_mipmaps(instance, physical_device, device, command_pool, graphics_queue, &vk_image, vk::Format::R8G8B8A8_SRGB, image.dimensions().0, image.dimensions().1, mip_levels);
        
        // (vk_image, device_memory, mip_levels)
    }

    // fn generate_mipmaps(instance: &Instance, physical_device: &PhysicalDevice, device: &Device, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, image: &vk::Image, image_format: vk::Format, width: u32, height: u32, mip_levels: u32) {
    //     let format_properties = unsafe {
    //         instance.get_physical_device_format_properties(*physical_device, image_format)
    //     };

    //     if !format_properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
    //         panic!("Texture image format does not support linear blitting!");
    //     }

    //     let command_buffer = Self::begin_single_time_command(device, command_pool);
        
    //     let mut image_barrier = vk::ImageMemoryBarrier {
    //         s_type: StructureType::IMAGE_MEMORY_BARRIER,
    //         image: *image,
    //         src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    //         dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    //         subresource_range: vk::ImageSubresourceRange {
    //             aspect_mask: vk::ImageAspectFlags::COLOR,
    //             base_array_layer: 0,
    //             layer_count: 1,
    //             level_count: 1,
    //             ..Default::default()
    //         },
    //         ..Default::default()
    //     };

    //     let mut mip_width = width as i32;
    //     let mut mip_height = height as i32;

    //     for i in 1..mip_levels {
    //         image_barrier.subresource_range.base_mip_level = i - 1;
    //         image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    //         image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    //         image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    //         image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

    //         unsafe {
    //             device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);
    //         }

    //         let blit = vk::ImageBlit {
    //             src_offsets: [
    //                 vk::Offset3D {
    //                     x: 0,
    //                     y: 0,
    //                     z: 0,
    //                 },
    //                 vk::Offset3D {
    //                     x: mip_width,
    //                     y: mip_height,
    //                     z: 1,
    //                 },
    //             ],
    //             src_subresource: vk::ImageSubresourceLayers {
    //                 aspect_mask: vk::ImageAspectFlags::COLOR,
    //                 mip_level: i - 1,
    //                 base_array_layer: 0,
    //                 layer_count: 1,
    //             },
    //             dst_offsets: [
    //                 vk::Offset3D {
    //                     x: 0,
    //                     y: 0,
    //                     z: 0,
    //                 },
    //                 vk::Offset3D {
    //                     x: if mip_width > 1 { mip_width / 2 } else { 1 },
    //                     y: if mip_height > 1 { mip_height / 2 } else { 1 },
    //                     z: 1,
    //                 },
    //             ],
    //             dst_subresource: vk::ImageSubresourceLayers {
    //                 aspect_mask: vk::ImageAspectFlags::COLOR,
    //                 mip_level: i,
    //                 base_array_layer: 0,
    //                 layer_count: 1,
    //             },
    //         };

    //         unsafe {
    //             device.cmd_blit_image(command_buffer, *image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, *image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
    //         }

    //         image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    //         image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    //         image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
    //         image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    //         unsafe {
    //             device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, DependencyFlags::empty(), &[], &[], &[image_barrier]);
    //         }

    //         if mip_width > 1 {
    //             mip_width /= 2;
    //         }
    //         if mip_height > 1 {
    //             mip_height /= 2;
    //         }
    //     }

    //     image_barrier.subresource_range.base_mip_level = mip_levels - 1;
    //     image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    //     image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    //     image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    //     image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    //     unsafe {
    //         device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, DependencyFlags::empty(), &[], &[], &[image_barrier]);
    //     } 

    //     Self::end_single_time_command(device, command_pool, graphics_queue, command_buffer);
    // }

    // fn create_image(instance: &Instance, physical_device: &PhysicalDevice, device: &Device, width: u32, height: u32, mip_levels: u32, num_samples: vk::SampleCountFlags, format: vk::Format, tiling: vk::ImageTiling, usage: vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags, allocator: &mut VkAllocator) -> (vk::Image, vk::DeviceMemory) {
    //     let image_info = vk::ImageCreateInfo {
    //         s_type: StructureType::IMAGE_CREATE_INFO,
    //         image_type: vk::ImageType::TYPE_2D,
    //         extent: vk::Extent3D {
    //             width,
    //             height,
    //             depth: 1,
    //         },
    //         mip_levels,
    //         array_layers: 1,
    //         format,
    //         tiling,
    //         initial_layout: vk::ImageLayout::UNDEFINED,
    //         usage,
    //         sharing_mode: vk::SharingMode::EXCLUSIVE,
    //         samples: num_samples,
    //         flags: vk::ImageCreateFlags::empty(),
    //         ..Default::default()
    //     };

    //     let image = unsafe {
    //         device.create_image(&image_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap();

    //     let mem_requirements = unsafe {
    //         device.get_image_memory_requirements(image)
    //     };

    //     let alloc_info = vk::MemoryAllocateInfo {
    //         s_type: StructureType::MEMORY_ALLOCATE_INFO,
    //         allocation_size: mem_requirements.size,
    //         memory_type_index: Self::find_memory_type(instance, physical_device, mem_requirements.memory_type_bits, properties).unwrap(),
    //         ..Default::default()
    //     };

    //     let image_memory = unsafe {
    //         device.allocate_memory(&alloc_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap();

    //     unsafe {
    //         device.bind_image_memory(image, image_memory, 0).unwrap();
    //     }

    //     (image, image_memory)
    // }

    // fn transition_image_layout(device: &Device, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, image: &vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32) {
    //     let command_buffer = Self::begin_single_time_command(device, command_pool);

    //     let mut barrier = vk::ImageMemoryBarrier {
    //         s_type: StructureType::IMAGE_MEMORY_BARRIER,
    //         old_layout,
    //         new_layout,
    //         src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    //         dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    //         image: *image,
    //         subresource_range: vk::ImageSubresourceRange {
    //             aspect_mask: vk::ImageAspectFlags::COLOR,
    //             base_mip_level: 0,
    //             level_count: mip_levels,
    //             base_array_layer: 0,
    //             layer_count: 1,
    //         },
    //         // src_access_mask: match old_layout { // This could cause issues, see transition barrier masks on https://vulkan-tutorial.com/Texture_mapping/Images
    //         //     vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
    //         //     vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
    //         //     _ => panic!("Unsupported layout transition!"),
    //         // },
    //         // dst_access_mask: match new_layout {
    //         //     vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
    //         //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
    //         //     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    //         //     _ => panic!("Unsupported layout transition!"),
    //         // },
    //         ..Default::default()
    //     };

    //     let (source_stage, destination_stage) = match (old_layout, new_layout) {
    //         (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
    //             barrier.src_access_mask = vk::AccessFlags::empty();
    //             barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    //             (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER)
    //         },
    //         (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
    //             barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    //             barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
    //             (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER)
    //         },
    //         //(vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS),
    //         _ => panic!("Unsupported layout transition! {} {}", old_layout.as_raw(), new_layout.as_raw()),
    //     };

    //     unsafe {
    //         device.cmd_pipeline_barrier(command_buffer, source_stage, destination_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
    //     }

    //     Self::end_single_time_command(device, command_pool, graphics_queue, command_buffer);
    // }

    // fn create_image_view(device: &Device, image: &vk::Image, format: vk::Format, aspect_flags: vk::ImageAspectFlags, mip_levels: u32, allocator: &mut VkAllocator) -> vk::ImageView {
    //     let view_info = vk::ImageViewCreateInfo {
    //         s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
    //         image: *image,
    //         view_type: vk::ImageViewType::TYPE_2D,
    //         format,
    //         subresource_range: vk::ImageSubresourceRange {
    //             aspect_mask: aspect_flags,
    //             base_mip_level: 0,
    //             level_count: mip_levels,
    //             base_array_layer: 0,
    //             layer_count: 1,
    //         },
    //         ..Default::default()
    //     };

    //     unsafe {
    //         device.create_image_view(&view_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap()
    // }

    fn create_texture_image_view(image_allocation: &mut AllocationInfo, mip_levels: u32, allocator: &mut VkAllocator) {
        allocator.create_image_view(image_allocation, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR, mip_levels).unwrap();
    }

    fn create_texture_sampler(device: &Device, instance: &Instance, physical_device: &PhysicalDevice, mip_levels: u32, allocator: &mut VkAllocator) -> vk::Sampler {
        let max_anisotropy = unsafe {
            instance.get_physical_device_properties(*physical_device).limits.max_sampler_anisotropy
        };
        let sampler_info = vk::SamplerCreateInfo {
            s_type: StructureType::SAMPLER_CREATE_INFO,
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            address_mode_u: vk::SamplerAddressMode::REPEAT,
            address_mode_v: vk::SamplerAddressMode::REPEAT,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::TRUE,
            max_anisotropy,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            mip_lod_bias: 0.0,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            ..Default::default()
        };

        unsafe {
            device.create_sampler(&sampler_info, Some(&allocator.get_allocation_callbacks())).unwrap()
        }
    }

    fn create_depth_resources(instance: &Instance, physical_device: &PhysicalDevice, swapchain_extent: &vk::Extent2D, msaa_samples: vk::SampleCountFlags, allocator: &mut VkAllocator) -> AllocationInfo {
        let depth_format = Self::find_depth_format(instance, physical_device);

        let mut allocation_info = allocator.create_image(swapchain_extent.width, swapchain_extent.height, 1, msaa_samples, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();

        // let (depth_image, depth_image_memory) = Self::create_image(instance, physical_device, device, swapchain_extent.width, swapchain_extent.height, 1, msaa_samples, depth_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL, allocator);

        allocator.create_image_view(&mut allocation_info, depth_format, vk::ImageAspectFlags::DEPTH, 1).unwrap();

        allocation_info
        // (depth_image, depth_image_memory, depth_image_view)
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

    fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
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
        //let (color_image, color_image_memory) = Self::create_image(instance, physical_device, device, swapchain_extent.width, swapchain_extent.height, 1, num_samples, swapchain_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL, allocator);

        let mut color_allocation = allocator.create_image(swapchain_extent.width, swapchain_extent.height, 1, num_samples, swapchain_format, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::MemoryPropertyFlags::DEVICE_LOCAL).unwrap();

        allocator.create_image_view(&mut color_allocation, swapchain_format, vk::ImageAspectFlags::COLOR, 1).unwrap();

        color_allocation
        // (color_image, color_image_memory, color_image_view)
    }

    fn load_model(path: &str) -> (Vec<Vertex>, Vec<u32>) {
        let (models, _) = tobj::load_obj(path, &tobj::LoadOptions::default()).unwrap();
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut unique_vertices: HashMap<Vertex, u32> = HashMap::new();

        for model in models {
            let mesh = model.mesh;
            for i in 0..mesh.indices.len() {
                let index = mesh.indices[i] as usize;
                let vertex = Vertex {
                    position: glm::vec3(mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]),
                    color: glm::vec3(1.0, 1.0, 1.0),
                    tex_coord: glm::vec2(mesh.texcoords[index * 2], 1.0 - mesh.texcoords[index * 2 + 1]),
                };
        
                if let hash_map::Entry::Vacant(e) = unique_vertices.entry(vertex) {
                    e.insert(vertices.len() as u32);
                    vertices.push(vertex);
                }
                indices.push(*unique_vertices.get(&vertex).unwrap());
            }
        }

        (vertices, indices)
    }

    fn find_memory_type(instance: &Instance, physical_device: &PhysicalDevice, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, Cow<'static, str>> {
        let mem_properties = unsafe {
            instance.get_physical_device_memory_properties(*physical_device)
        };

        for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
            if type_filter & (1 << i) != 0 && mem_type.property_flags.contains(properties) {
                return Ok(i as u32);
            }
        }
        Err(Cow::from("Failed to find suitable memory type!"))
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