use std::{borrow::Cow, collections::{hash_map, HashMap}, fmt::Formatter, path::PathBuf, sync::Arc};

use ash::vk::{self, CommandPool, Queue};
use image::DynamicImage;
use nalgebra_glm as glm;

use crate::{pipeline_manager::{GraphicsResource, GraphicsResourceType, PipelineConfig, ShaderInfo, Vertex}, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable, VkAllocator}};

macro_rules! free_allocations_add_error_string {
    ($allocator: expr, $allocations: expr, $error_string: expr) => {
        for allocation in $allocations {
            let error = $allocator.free_memory_allocation(allocation);
            if let Err(err) = error {
                $error_string.push_str(&format!("\n{}", err));
            }
        }
    };
}

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

#[derive(Clone)]
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
        GraphicsResourceType::UniformBuffer(self.buffer.to_u8())
    }
}

pub struct TextureResource {
    pub image: DynamicImage,
    pub binding: u32,
    pub stage: vk::ShaderStageFlags,
}

impl GraphicsResource for TextureResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: self.stage,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> GraphicsResourceType {
        GraphicsResourceType::Texture(self.image.clone())
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
    fn get_shader_infos(&self) -> Vec<ShaderInfo>;
    fn get_msaa_samples(&self) -> vk::SampleCountFlags;
}

pub struct ObjectToRender<T: Vertex> {
    vertex_allocation: AllocationInfo,
    index_allocation: AllocationInfo,
    extra_resource_allocations: Vec<(vk::DescriptorSetLayoutBinding, AllocationInfo)>,
    pipeline_config: PipelineConfig,
    original_object: Arc<dyn GraphicsObject<T>>,
}

impl<T: Vertex + Clone + 'static> ObjectToRender<T> {
    pub fn new(original_object: Arc<dyn GraphicsObject<T>>, swapchain_format: vk::Format, depth_format: vk::Format, command_pool: &CommandPool, graphics_queue: &Queue,allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let vertices = original_object.get_vertices();
        let vertex_data = vertices.iter().map(|v| v.to_u8()).flatten().collect::<Vec<u8>>();
        let vertex_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &vertex_data, vk::BufferUsageFlags::VERTEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => return Err(Cow::from(e)),
        };
        let indices = original_object.get_indices();
        let index_data = indices.iter().map(|i| i.to_ne_bytes()).flatten().collect::<Vec<u8>>();
        let index_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &index_data, vk::BufferUsageFlags::INDEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                free_allocations_add_error_string!(allocator, vec![vertex_allocation], error_str);
                return Err(Cow::from(error_str));
            },
        
        };

        let mut extra_resource_allocations = Vec::with_capacity(original_object.get_resources().len());
        for resource in original_object.get_resources() {
            match resource.get_resource() {
                GraphicsResourceType::UniformBuffer(buffer) => {
                    let allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &buffer, vk::BufferUsageFlags::UNIFORM_BUFFER, false) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            free_allocations_add_error_string!(allocator, vec![vertex_allocation, index_allocation], error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    extra_resource_allocations.push((resource.get_descriptor_set_layout_binding(), allocation));
                }
                GraphicsResourceType::Texture(image) => {
                    let allocation = match allocator.create_device_local_image(image, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1, false) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            free_allocations_add_error_string!(allocator, vec![vertex_allocation, index_allocation], error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    extra_resource_allocations.push((resource.get_descriptor_set_layout_binding(), allocation));
                }
            }
        }

        let vertex_sample = match vertices.first() {
            Some(v) => v.clone(),
            None => return Err("No vertices found when trying to create graphics object for rendering".into()),
        };

        let pipeline_config = PipelineConfig::new(
            original_object.get_shader_infos(),
            vertex_sample.get_input_binding_description(),
            vertex_sample.get_attribute_descriptions(),
            original_object.get_resources(),
            original_object.get_msaa_samples(),
            swapchain_format,
            depth_format,
        );

        Err("Not implemented".into())
    }
}