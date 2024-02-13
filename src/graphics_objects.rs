use std::{borrow::Cow, fmt::Formatter, path::PathBuf, sync::Arc};

use ash::vk::{self, CommandPool, Queue};
use nalgebra_glm as glm;

use crate::{pipeline_manager::{GraphicsResource, GraphicsResourceType, PipelineConfig, Vertex}, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable, VkAllocator}};

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}

impl Serializable for UniformBufferObject {
    fn to_u8(&self) -> Vec<u8> {
        let model = self.model.as_slice();
        let view = self.view.as_slice();
        let proj = self.proj.as_slice();
        let mut result = Vec::with_capacity(std::mem::size_of::<UniformBufferObject>());
        for i in 0..16 {
            result.extend_from_slice(&model[i].to_ne_bytes());
        }
        for i in 0..16 {
            result.extend_from_slice(&view[i].to_ne_bytes());
        }
        for i in 0..16 {
            result.extend_from_slice(&proj[i].to_ne_bytes());
        }

        result
    }
}

pub struct UniformBufferResource<T: Clone> {
    pub buffer: T,
    pub binding: u32,
}

impl<T: Clone + Serializable> GraphicsResource for UniformBufferResource<T> {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> crate::pipeline_manager::GraphicsResourceType {
        GraphicsResourceType::UniformBuffer(Box::new(self.buffer.clone()))
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

    fn get_resource(&self) -> GraphicsResourceType {
        GraphicsResourceType::Texture(image::open(self.path.clone()).unwrap())
    }
}

pub trait GraphicsObject<T: Vertex> {
    fn get_vertices(&self) -> Vec<T>;
    fn get_indices(&self) -> Vec<u32>;
    fn get_resources(&self) -> Vec<Arc<dyn GraphicsResource>>;
}

pub struct ObjectToRender<T: Vertex> {
    vertex_allocation: AllocationInfo,
    index_allocation: AllocationInfo,
    extra_resource_allocations: Vec<(vk::DescriptorSetLayoutBinding, AllocationInfo)>,
    pipeline_config: PipelineConfig,
    original_object: Arc<dyn GraphicsObject<T>>,
}

impl<T: Vertex> ObjectToRender<T> {
    pub fn new(original_object: Arc<dyn GraphicsObject<T>>, command_pool: &CommandPool, graphics_queue: &Queue,allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let vertex_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &original_object.get_vertices(), vk::BufferUsageFlags::VERTEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => return Err(Cow::from(e)),
        };
        let index_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &original_object.get_indices(), vk::BufferUsageFlags::INDEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                todo!();
                let error = allocator.free_memory_allocation(vertex_allocation);
                return Err(Cow::from(e));
            },
        
        };

        let mut extra_resource_allocations = Vec::with_capacity(original_object.get_resources().len());
        
    }
}
