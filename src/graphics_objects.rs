use std::{fmt::Formatter, path::PathBuf};

use ash::vk;
use nalgebra_glm as glm;

use crate::{pipeline_manager::GraphicsResource, vertex::SimpleVertex, vk_allocator::Serializable};

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}

impl GraphicsResource for UniformBufferObject {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}