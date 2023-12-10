use std::{collections::HashSet, fs::read_to_string};

use ash::{Entry, Instance, vk::{SurfaceKHR, PhysicalDevice, DeviceQueueCreateInfo, DeviceCreateInfo, Queue, SwapchainCreateInfoKHR, ImageView, ImageViewCreateInfo, StructureType, InstanceCreateFlags, InstanceCreateInfo, KhrPortabilityEnumerationFn, self, DebugUtilsMessengerCreateInfoEXT, SwapchainKHR, Image, ComponentMapping, ImageSubresourceRange, Offset2D}, Device, extensions::{khr::{Swapchain, Surface}, ext::DebugUtils}};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle, HasDisplayHandle, HasWindowHandle};
use shaderc::{Compiler, ShaderKind};
use winit::window::Window;



#[cfg(debug_assertions)]
const IS_DEBUG_MODE: bool = true;
#[cfg(debug_assertions)]
const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

#[cfg(not(debug_assertions))]
const IS_DEBUG_MODE: bool = false;

pub struct VkAndWindowController {
    window: Window,
    entry: Entry,
    instance: Instance,
    debug_messenger_create_info: Option<DebugUtilsMessengerCreateInfoEXT>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    device: Device,
    graphics_queue: Queue,
    present_queue: Queue,
    surface: SurfaceKHR,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<ImageView>,
    pipeline_layout: vk::PipelineLayout,
}

impl VkAndWindowController {
    const DEVICE_EXTENSIONS: [*const i8; 1] = [Swapchain::name().as_ptr()];

    pub fn new(window: Window, application_name: &str) -> Self {
        let entry = Entry::linked();
        
        let debug_messenger_create_info = if IS_DEBUG_MODE {
            Some(Self::get_debug_messenger_create_info())
        } else {
            None
        };
        let instance = Self::create_instance(&entry, application_name, &window, debug_messenger_create_info.as_ref());

        
        let mut debug_messenger = None;
        if IS_DEBUG_MODE {
            debug_messenger = Some(Self::setup_debug_messenger(&entry, &instance, debug_messenger_create_info.unwrap()));
        }

        let surface = Self::create_surface(&entry, &instance, &window);
        
        let physical_device = Self::pick_physical_device(&entry, &instance, &surface);

        let queue_families = Self::find_queue_families(&entry, &instance, &physical_device, &surface);

        let device = Self::create_logical_device(&entry, &instance, &physical_device, &surface);

        let (graphics_queue, present_queue) = Self::create_graphics_and_present_queue(&device, &queue_families);

        let swapchain = Self::create_swapchain(&entry, &instance, &physical_device, &device, &surface, &window);

        let swapchain_images = Self::get_swapchain_images(&instance, &device, &swapchain);

        let swapchain_image_format = Self::choose_swap_surface_format(&Self::query_swapchain_support(&entry, &instance, &physical_device, &surface).formats).format;

        let swapchain_extent = Self::choose_swap_extent(&Self::query_swapchain_support(&entry, &instance, &physical_device, &surface).capabilities, &window);

        let swapchain_image_views = Self::create_image_views(&instance, &device, &swapchain_images, swapchain_image_format);

        let pipeline_layout = Self::create_pipeline_layout(&device);

        Self {
            window,
            entry,
            instance,
            debug_messenger_create_info,
            debug_messenger,
            device,
            graphics_queue,
            present_queue,
            surface,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
            pipeline_layout,
        }
    }
}

// Instance
impl VkAndWindowController {
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
        println!("Adding KhrPortabilityEnumerationFn here might not work!");
        required_instance_extensions.push(KhrPortabilityEnumerationFn::name().as_ptr());
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

        create_info.flags |= InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;

        if IS_DEBUG_MODE {
            let validation_layers = VALIDATION_LAYERS;

            create_info.enabled_layer_count = validation_layers.len() as u32;
            create_info.pp_enabled_layer_names = validation_layers.as_ptr().cast();
            
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

        let validation_layers = VALIDATION_LAYERS;

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
}

// Debug messenger
impl VkAndWindowController {
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


// Physical device
impl VkAndWindowController {
    fn pick_physical_device(entry: &Entry, instance: &Instance, surface: &SurfaceKHR) -> PhysicalDevice{
        let mut device_vec = unsafe {
            instance.enumerate_physical_devices()
        }.expect("Expected to be able to look for physical devices (GPU)!");

        if device_vec.is_empty() {
            panic!("No physical devices found that support Vulkan!");
        }

        device_vec.sort_by_key(|device| Self::rate_physical_device_suitability(instance, device));
        device_vec.reverse();

        for device in device_vec.iter() {
            if Self::is_device_suitable(entry, instance, device, surface) {
                return *device;
            }
        }

        panic!("No suitable physical device found!");
    }

    fn is_device_suitable(entry: &Entry, instance: &Instance, device: &PhysicalDevice, surface: &SurfaceKHR) -> bool {
        let indices = Self::find_queue_families(entry, instance, device, surface);
        let swapchain_support = Self::query_swapchain_support(entry, instance, device, surface);

        indices.is_complete() && Self::check_device_extension_support(instance, device) && Self::is_swapchain_adequate(&swapchain_support)
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
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

// Queues
impl VkAndWindowController {
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
}

// Logical device
impl VkAndWindowController {
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
}

// Surface
impl VkAndWindowController {
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
}

struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

// Swapchain
impl VkAndWindowController {
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

    fn create_swapchain(entry: &Entry, instance: &Instance, physical_device: &PhysicalDevice, device: &Device, surface: &SurfaceKHR, window: &Window) -> SwapchainKHR {
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
            Swapchain::new(instance, device).create_swapchain(&swapchain_create_info, None)
        }.unwrap()
    }

    fn get_swapchain_images(instance: &Instance, device: &Device, swapchain: &SwapchainKHR) -> Vec<Image> {
        unsafe {
            Swapchain::new(instance, device).get_swapchain_images(*swapchain)
        }.unwrap()
    }

    fn create_image_views(instance: &Instance, device: &Device, swapchain_images: &Vec<Image>, swapchain_image_format: vk::Format) -> Vec<ImageView> {
        let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());

        for swapchain_image in swapchain_images.iter() {
            let create_info = ImageViewCreateInfo {
                s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
                image: *swapchain_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: swapchain_image_format,
                components: ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            swapchain_image_views.push(unsafe {
                device.create_image_view(&create_info, None)
            }.unwrap());
        }

        swapchain_image_views
    }
}

// Graphics pipeline
impl VkAndWindowController {
    fn create_graphics_pipeline(device: &Device, swapchain_extent: &vk::Extent2D, pipeline_layout: &vk::PipelineLayout) {
        let vert_shader_code = Self::compile_shader("../assets/shaders/triangle.vert", ShaderKind::Vertex, "triangle.vert");
        let frag_shader_code = Self::compile_shader("../assets/shaders/triangle.frag", ShaderKind::Fragment, "triangle.frag");

        let vert_shader_module = Self::create_shader_module(device, vert_shader_code);
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code);

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

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];
        
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: std::ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: std::ptr::null(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: swapchain_extent.width as f32,
            height: swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: Offset2D {
                x: 0,
                y: 0,
            },
            extent: *swapchain_extent,
        };

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            s_type: StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

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
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            s_type: StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 1.0,
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


        // This should always happen at the end
        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }
    }

    fn compile_shader(path: &str, shader_kind: ShaderKind, identifier: &str) -> Vec<u32> {
        let compiler = Compiler::new().unwrap();
        let artifact = compiler.compile_into_spirv(&read_to_string(path).unwrap(), shader_kind, identifier, "main", None).unwrap();
        artifact.as_binary().to_owned()
    }

    fn create_shader_module(device: &Device, code: Vec<u32>) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: code.len() * std::mem::size_of::<u32>(),
            p_code: code.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device.create_shader_module(&create_info, None)
        }.unwrap()
    }

    fn create_pipeline_layout(device: &Device) -> vk::PipelineLayout {
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 0,
            p_set_layouts: std::ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            ..Default::default()
        };

        unsafe {
            device.create_pipeline_layout(&pipeline_layout_create_info, None)
        }.unwrap()
    }
}