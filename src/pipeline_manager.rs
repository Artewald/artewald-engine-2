use std::{borrow::Cow, ffi::CString, fs::read_to_string, hash::{Hash, Hasher}};

use ash::{vk::{self, DescriptorSetLayoutBinding, RenderPass, SampleCountFlags, StructureType, VertexInputAttributeDescription, VertexInputBindingDescription}, Device};
use image::DynamicImage;
use shaderc::{Compiler, ShaderKind};

use crate::vk_allocator::{Serializable, VkAllocator};

pub enum GraphicsResourceType {
    UniformBuffer(Vec<u8>),
    DynamicUniformBuffer(Vec<u8>),
    Texture(DynamicImage),
}

pub trait Vertex: Serializable + Hash + Clone + 'static {
    fn get_input_binding_description(&self) -> vk::VertexInputBindingDescription;
    fn get_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait GraphicsResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding;
    fn get_resource(&self) -> GraphicsResourceType;
}

#[derive(PartialEq, Eq, Clone)]
pub struct ShaderInfo {
    pub path: std::path::PathBuf,
    pub shader_stage_flag: vk::ShaderStageFlags,
    pub entry_point: CString,
}

#[derive(Clone)]
pub struct PipelineConfig {
    shaders: Vec<ShaderInfo>,
    vertex_binding_info: vk::VertexInputBindingDescription,
    vertex_attribute_info: Vec<vk::VertexInputAttributeDescription>,
    msaa_samples: vk::SampleCountFlags,
    swapchain_format: vk::Format,
    depth_format: vk::Format,
    descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pipeline_layout: Option<vk::PipelineLayout>,
}

impl PipelineConfig {
    pub fn new(device: &Device, shaders: Vec<ShaderInfo>, vertex_binding_info: VertexInputBindingDescription, vertex_attribute_info: Vec<VertexInputAttributeDescription>, descriptor_set_layout_bindings: &[DescriptorSetLayoutBinding], msaa_samples: vk::SampleCountFlags, swapchain_format: vk::Format, depth_format: vk::Format, allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        if vertex_attribute_info.is_empty() {
            return Err(Cow::Borrowed("Vertex attribute descriptions are empty"));
        }
        if vertex_attribute_info.iter().any(|attribute| attribute.binding != vertex_binding_info.binding) {
            return Err(Cow::Borrowed("Vertex attribute descriptions have different binding than the vertex input binding description"));
        }
        // Check if any of the vertex attribute descriptions have the same location
        for i in 0..vertex_attribute_info.len() {
            for j in i + 1..vertex_attribute_info.len() {
                if vertex_attribute_info[i].location == vertex_attribute_info[j].location {
                    return Err(Cow::Borrowed("Vertex attribute descriptions have the same location"));
                }
            }
        }

        Ok(PipelineConfig {
            shaders,
            vertex_binding_info,
            vertex_attribute_info,
            msaa_samples,
            swapchain_format,
            depth_format,
            descriptor_set_layout_bindings: descriptor_set_layout_bindings.to_vec(),
            descriptor_set_layout: None,
            pipeline_layout: None,
        })
    }

    fn create_graphics_pipeline(&mut self, device: &Device, swapchain_extent: &vk::Extent2D, render_pass: RenderPass, allocator: &mut VkAllocator) -> Result<vk::Pipeline, Cow<'static, str>> {
        for shader in self.shaders.iter() {
            if !(shader.shader_stage_flag == vk::ShaderStageFlags::VERTEX ||
                shader.shader_stage_flag == vk::ShaderStageFlags::FRAGMENT)  
             {
                 return Err(format!("The shader stage flag for shader with path {:?} cannot be more or less than one constant!", shader.path).into());
             };   
        }

        let shader_modules: Vec<(ShaderInfo, vk::ShaderModule)> = self.shaders.iter().map(|shader_info| {
            let shader_kind = match shader_info.shader_stage_flag {
                vk::ShaderStageFlags::VERTEX => ShaderKind::Vertex,
                vk::ShaderStageFlags::FRAGMENT => ShaderKind::Fragment,
                _ => panic!("Invalid shader stage flag for shader with path {:?}. This should never happen! The stage flag had number: {}!", shader_info.path, shader_info.shader_stage_flag.as_raw()),
            };
            let code = Self::compile_shader(&shader_info.path, shader_info.entry_point.to_str().unwrap(), shader_kind, &shader_info.path.to_string_lossy());
            let module = Self::create_shader_module(device, code, allocator);
            (shader_info.clone(), module)
        }).collect::<Vec<_>>();

        let shader_stage_create_infos: Vec<vk::PipelineShaderStageCreateInfo> = shader_modules.iter().map(|(shader_info, shader_module)| {
            vk::PipelineShaderStageCreateInfo {
                s_type: StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage: shader_info.shader_stage_flag,
                module: *shader_module,
                p_name: shader_info.entry_point.as_ptr(),
                ..Default::default()
            }
        }).collect();

        let binding_description = self.vertex_binding_info;
        let attribute_descriptions = self.vertex_attribute_info.clone();

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
            polygon_mode: vk::PolygonMode::FILL,//LINE,//
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
            rasterization_samples: self.msaa_samples,
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

        let descriptor_set_layout = self.descriptor_set_layout;
        let pipeline_layout = self.get_or_create_pipeline_layout(device, allocator);

        // let render_pass = self.create_render_pass(device, allocator);

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            stage_count: shader_stage_create_infos.len() as u32,
            p_stages: shader_stage_create_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], Some(&allocator.get_allocation_callbacks()))
        }.unwrap()[0];

        for (_, shader_module) in shader_modules {
            unsafe {
                device.destroy_shader_module(shader_module, Some(&allocator.get_allocation_callbacks()));
            }
        }

        Ok(graphics_pipeline)
    }

    fn compile_shader(path: &std::path::PathBuf, entry_point_name: &str, shader_kind: ShaderKind, identifier: &str) -> Vec<u32> {
        let compiler = Compiler::new().unwrap();
        let artifact = compiler.compile_into_spirv(&read_to_string(path).unwrap(), shader_kind, identifier, entry_point_name, None).unwrap();
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

    fn get_or_create_pipeline_layout(&mut self, device: &Device, allocator: &mut VkAllocator) -> vk::PipelineLayout {
        if self.pipeline_layout.is_some() {
            return self.pipeline_layout.unwrap();
        }

        let descriptor_set_layouts = [self.get_or_create_descriptor_set_layout(device, allocator)];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: 1,
            p_set_layouts: descriptor_set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            ..Default::default()
        };
        self.pipeline_layout = Some(unsafe {
            device.create_pipeline_layout(&pipeline_layout_create_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap());
        self.pipeline_layout.unwrap()
    }

    fn get_or_create_descriptor_set_layout(&mut self, device: &Device, allocator: &mut VkAllocator) -> vk::DescriptorSetLayout {
        if self.descriptor_set_layout.is_some() {
            return self.descriptor_set_layout.unwrap();
        }
        
        let layout_bindings = self.descriptor_set_layout_bindings.clone();

        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: layout_bindings.len() as u32,
            p_bindings: layout_bindings.as_ptr(),
            ..Default::default()
        };

        self.descriptor_set_layout = Some(unsafe {
            device.create_descriptor_set_layout(&layout_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap());

        self.descriptor_set_layout.unwrap()
    }

    pub fn borrow_descriptor_set_layout(&self) -> Option<&vk::DescriptorSetLayout> {
        self.descriptor_set_layout.as_ref()
    }

    pub fn get_pipeline_layout(&self) -> Option<vk::PipelineLayout> {
        self.pipeline_layout
    }
}

impl Eq for PipelineConfig {}

impl PartialEq for PipelineConfig {
    fn eq(&self, other: &Self) -> bool {
        self.shaders == other.shaders &&
        self.vertex_binding_info.binding == other.vertex_binding_info.binding &&
        self.vertex_binding_info.stride == other.vertex_binding_info.stride &&
        self.vertex_binding_info.input_rate == other.vertex_binding_info.input_rate &&
        self.vertex_attribute_info.iter().all(|attribute| other.vertex_attribute_info.iter().any(|other_attribute| attribute.binding == other_attribute.binding && attribute.location == other_attribute.location && attribute.format == other_attribute.format && attribute.offset == other_attribute.offset)) &&
        self.msaa_samples == other.msaa_samples &&
        self.swapchain_format == other.swapchain_format &&
        self.depth_format == other.depth_format &&
        self.descriptor_set_layout_bindings.iter().all(|binding| other.descriptor_set_layout_bindings.iter().any(|binding2| {
            binding.binding == binding2.binding &&
            binding.descriptor_type == binding2.descriptor_type &&
            binding.descriptor_count == binding2.descriptor_count &&
            binding.stage_flags == binding2.stage_flags &&
            binding.p_immutable_samplers == binding2.p_immutable_samplers
        })) &&
        self.descriptor_set_layout_bindings.len() == other.descriptor_set_layout_bindings.len() //&&
        // self.descriptor_set_layout == other.descriptor_set_layout
    }
}

impl Hash for PipelineConfig {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shaders.iter().for_each(|shader| shader.path.hash(state));
        self.vertex_binding_info.binding.hash(state);
        self.vertex_binding_info.stride.hash(state);
        self.vertex_binding_info.input_rate.hash(state);
        self.vertex_attribute_info.iter().for_each(|attribute| {
            attribute.location.hash(state);
            attribute.binding.hash(state);
            attribute.format.hash(state);
            attribute.offset.hash(state);
        });
        self.msaa_samples.hash(state);
        self.swapchain_format.hash(state);
        self.depth_format.hash(state);
        self.descriptor_set_layout_bindings.iter().for_each(|binding| {
            binding.binding.hash(state);
            binding.descriptor_type.hash(state);
            binding.descriptor_count.hash(state);
            binding.stage_flags.hash(state);
            binding.p_immutable_samplers.hash(state);
        });
        self.descriptor_set_layout_bindings.len().hash(state);
    }
}

pub struct PipelineManager {
    graphics_pipelines: Vec<(PipelineConfig, vk::Pipeline)>,
    render_pass: Option<vk::RenderPass>,
}

impl PipelineManager {
    pub fn new(device: &Device, swapchain_format: vk::Format, msaa_samples: SampleCountFlags, depth_format: vk::Format, allocator: &mut VkAllocator) -> Self {
        PipelineManager {
            graphics_pipelines: Vec::new(),
            render_pass: Some(Self::create_render_pass(device, swapchain_format, msaa_samples, depth_format, allocator)),
        }
    }

    pub fn get_or_create_pipeline(&mut self, pipeline_config: &mut PipelineConfig, device: &Device, swapchain_extent: &vk::Extent2D, allocator: &mut VkAllocator) -> Result<vk::Pipeline, Cow<'static, str>> {
        if let Some((p_config, pipeline)) = self.graphics_pipelines.iter().find(|(config, _)| config == pipeline_config) {
            if pipeline_config.pipeline_layout.is_none() {
                // This is needed because some new objects with the same pipeline layout might be added, so we need to update their pipeline layout and descriptor_set_layout
                pipeline_config.pipeline_layout = Some(p_config.pipeline_layout.unwrap());
                pipeline_config.descriptor_set_layout = Some(p_config.descriptor_set_layout.unwrap());
            }
            Ok(*pipeline)
        } else {
            println!("Did not find the pipeline in the list, creating a new one");
            let pipeline = pipeline_config.create_graphics_pipeline(device, swapchain_extent, self.render_pass.unwrap(), allocator)?;
            self.graphics_pipelines.push((pipeline_config.clone(), pipeline));
            Ok(pipeline)
        }
    }

    pub fn destroy(&mut self, device: &Device, allocator: &mut VkAllocator) {
        for (config, pipeline) in self.graphics_pipelines.iter() {
            unsafe {
                device.destroy_pipeline(*pipeline, Some(&allocator.get_allocation_callbacks()));
                device.destroy_pipeline_layout(config.pipeline_layout.unwrap(), Some(&allocator.get_allocation_callbacks()));
                device.destroy_descriptor_set_layout(config.descriptor_set_layout.unwrap(), Some(&allocator.get_allocation_callbacks()));
                // device.destroy_descriptor_set_layout(config.descriptor_set_layout.unwrap(), Some(&allocator.get_allocation_callbacks()));
            }
        }
        unsafe {
            device.destroy_render_pass(self.render_pass.unwrap(), Some(&allocator.get_allocation_callbacks()));
        }
        self.graphics_pipelines.clear();
    }

    pub fn get_render_pass(&self) -> Option<vk::RenderPass> {
        self.render_pass
    }

    fn create_render_pass(device: &Device, swapchain_format: vk::Format, msaa_samples: SampleCountFlags, depth_format: vk::Format, allocator: &mut VkAllocator) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_format,
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
            format: depth_format,
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
            format: swapchain_format,
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
}