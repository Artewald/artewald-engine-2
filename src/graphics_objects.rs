use std::{fmt::Formatter, path::PathBuf};

use ash::vk;
use nalgebra_glm as glm;

use crate::{pipeline_manager::{GraphicsResource, PipelineConfig}, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable}};

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}

pub struct UniformBufferResource<T> {
    pub buffer: T,
    pub binding: u32,
}

impl<T> GraphicsResource for UniformBufferResource<T> {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimpleObjectTextureResource {
    pub path: PathBuf,
    pub binding: u32,
}

impl GraphicsResource for SimpleObjectTextureResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        }
    }
}

pub struct GraphicsObject {
    vertex_allocation: AllocationInfo,
    index_allocation: AllocationInfo,
    extra_resource_allocations: Vec<(vk::DescriptorSetLayoutBinding, AllocationInfo)>,
    pipeline_config: PipelineConfig,
}

