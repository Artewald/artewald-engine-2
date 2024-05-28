use std::{borrow::Cow, collections::{hash_map, HashMap}, fmt::Formatter, path::PathBuf, sync::{Arc, RwLock}, time::Instant};

use ash::{vk::{self, CommandPool, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, PhysicalDevice, Queue, Sampler, StructureType, WriteDescriptorSet}, Device, Instance};
use image::DynamicImage;
use nalgebra_glm as glm;

use crate::{pipeline_manager::{ObjectInstanceGraphicsResource, ObjectInstanceGraphicsResourceType, ObjectTypeGraphicsResource, ObjectTypeGraphicsResourceType, PipelineConfig, PipelineManager, ShaderInfo, Vertex}, sampler_manager::{SamplerConfig, SamplerManager}, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable, VkAllocator}, vk_controller::{self, IndexAllocation, VertexAllocation, VerticesIndicesHash, VkController}};

#[macro_export]
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

type ResourceAllocation = (ResourceID, vk::DescriptorSetLayoutBinding, AllocationInfo, DescriptorType, Option<Sampler>);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Copy, Clone)]
pub struct ResourceID(pub u32);

#[derive(Clone)]
pub struct UniformBufferResource<T: Clone> {
    pub buffer: T,
    pub binding: u32,
}

#[derive(Clone)]
pub struct DynamicUniformBufferResource<T: Clone> {
    pub buffer: T,
    pub binding: u32,
}

impl<T: Clone + Serializable> ObjectTypeGraphicsResource for UniformBufferResource<T> {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> crate::pipeline_manager::ObjectTypeGraphicsResourceType {
        ObjectTypeGraphicsResourceType::UniformBuffer(self.buffer.to_u8())
    }
}

impl<T:Clone + Serializable> ObjectInstanceGraphicsResource for UniformBufferResource<T> {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> crate::pipeline_manager::ObjectInstanceGraphicsResourceType {
        ObjectInstanceGraphicsResourceType::DynamicUniformBuffer(self.buffer.to_u8())
    }
}

pub struct TextureResource {
    pub image: DynamicImage,
    pub binding: u32,
    pub stage: vk::ShaderStageFlags,
    // pub sampler: Sampler,
}

impl ObjectTypeGraphicsResource for TextureResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: self.stage,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> ObjectTypeGraphicsResourceType {
        ObjectTypeGraphicsResourceType::Texture(self.image.clone())
    }
}

pub trait GraphicsObject<T: Vertex> {
    fn get_vertices(&self) -> Vec<T>;
    fn get_indices(&self) -> Vec<u32>;
    fn get_instance_resources(&self) -> Vec<(ResourceID, Arc<RwLock<dyn ObjectInstanceGraphicsResource>>)>;
    fn get_shader_infos(&self) -> Vec<ShaderInfo>;
    fn get_vertices_and_indices_hash(&self) -> VerticesIndicesHash;
    fn get_type_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectTypeGraphicsResource + 'static)>>)>;
}

pub trait Renderable {
    fn get_vertices_and_indices_hash(&self) -> VerticesIndicesHash;
    fn get_vertex_byte_data(&self) -> Vec<u8>;
    fn get_indices(&self) -> Vec<u32>;
    fn get_object_instance_resources(&self) -> Vec<(ResourceID, Arc<RwLock<dyn ObjectInstanceGraphicsResource>>)>;
    fn get_vertex_binding_info(&self) -> vk::VertexInputBindingDescription;
    fn get_vertex_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription>;
    fn get_shader_infos(&self) -> Vec<ShaderInfo>;
    fn get_type_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectTypeGraphicsResource + 'static)>>)>;
}

impl<T: Vertex> Renderable for Arc<RwLock<dyn GraphicsObject<T>>> {
    fn get_vertices_and_indices_hash(&self) -> VerticesIndicesHash {
        self.read().unwrap().get_vertices_and_indices_hash()
    }
    
    fn get_vertex_byte_data(&self) -> Vec<u8> {
        let original_object_locked = self.read().unwrap();
        let vertices = original_object_locked.get_vertices();
        let vertex_data = vertices.iter().map(|v| v.to_u8()).flatten().collect::<Vec<u8>>();
        vertex_data
    }
    
    fn get_indices(&self) -> Vec<u32> {
        self.read().unwrap().get_indices()
    }
    
    fn get_object_instance_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectInstanceGraphicsResource + 'static)>>)> {
        self.read().unwrap().get_instance_resources()
    }
    
    fn get_shader_infos(&self) -> Vec<ShaderInfo> {
        self.read().unwrap().get_shader_infos()
    }
    
    fn get_vertex_binding_info(&self) -> vk::VertexInputBindingDescription {
        T::get_input_binding_description()
    }
    
    fn get_vertex_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
        T::get_attribute_descriptions()
    }
    
    fn get_type_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectTypeGraphicsResource + 'static)>>)> {
        self.read().unwrap().get_type_resources()
    }
    
    
}