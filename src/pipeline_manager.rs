use std::borrow::Cow;

use ash::{vk::{self, StructureType}, Device};

use crate::vk_allocator::VkAllocator;

pub trait Vertex {
    fn vertex_input_binding_descriptions(&self) -> vk::VertexInputBindingDescription;
    fn get_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait GraphicsResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding;
}

pub struct ShaderInfo {
    pub path: std::path::PathBuf,
    pub shader_stage_create_info: vk::PipelineShaderStageCreateInfo,
    pub entry_point: String,
}

pub struct PipelineConfig {
    shaders: Vec<ShaderInfo>,
    vertex_binding_info: vk::VertexInputBindingDescription,
    vertex_attribute_info: Vec<vk::VertexInputAttributeDescription>,
    descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl PipelineConfig {
    pub fn new(shaders: Vec<ShaderInfo>, vertex: Box<dyn Vertex>, resources: Option<Vec<Box<dyn GraphicsResource>>>) -> Result<Self, Cow<'static, str>> {
        let vertex_binding_info = vertex.vertex_input_binding_descriptions();
        let vertex_attribute_info = vertex.get_attribute_descriptions();
        if vertex_attribute_info.len() == 0 {
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

        let mut descriptor_set_layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
        if let Some(resources) = resources {
            for resource in resources {
                let resource_binding = resource.get_descriptor_set_layout_binding();
                if descriptor_set_layout_bindings.iter().any(|binding| binding.binding == resource_binding.binding) {
                    return Err(Cow::Borrowed("Descriptor set layout binding with the same binding already exists"));
                }
                descriptor_set_layout_bindings.push(resource_binding);
            }
        }

        Ok(PipelineConfig {
            shaders,
            vertex_binding_info,
            vertex_attribute_info,
            descriptor_set_layout_bindings,
        })
    }

    pub fn create_descriptor_set_layout(&self, device: &Device, allocator: &mut VkAllocator) -> vk::DescriptorSetLayout {
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            binding_count: self.descriptor_set_layout_bindings.len() as u32,
            p_bindings: self.descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device.create_descriptor_set_layout(&layout_info, Some(&allocator.get_allocation_callbacks()))
        }.unwrap()
    }
}

pub struct PipelineManager {
    
}
