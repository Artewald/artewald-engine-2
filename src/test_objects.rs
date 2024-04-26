use std::{collections::{hash_map, HashMap}, sync::{Arc, RwLock}};
use ash::{vk::{DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, StructureType}, Device};
use nalgebra_glm as glm;

use crate::{graphics_objects::{GraphicsObject, TextureResource, UniformBufferObject, UniformBufferResource}, pipeline_manager::{GraphicsResource, GraphicsResourceType, ShaderInfo}, vertex::{OnlyTwoDPositionVertex, SimpleVertex}, vk_allocator::{Serializable, VkAllocator}};

pub struct SimpleRenderableObject {
    pub vertices: Vec<SimpleVertex>,
    pub indices: Vec<u32>,
    pub uniform_buffer: Arc<RwLock<UniformBufferResource<UniformBufferObject>>>,
    pub texture: Arc<RwLock<TextureResource>>,
    pub shaders: Vec<ShaderInfo>,
    pub descriptor_set_layout: Option<DescriptorSetLayout>,
}       

impl GraphicsObject<SimpleVertex> for SimpleRenderableObject {
    fn get_vertices(&self) -> Vec<SimpleVertex> {
        self.vertices.clone()
    }

    fn get_indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    fn get_resources(&self) -> Vec<(u32, Arc<RwLock<(dyn GraphicsResource + 'static)>>)> {
        vec![
            (1, self.uniform_buffer.clone()),
            (2, self.texture.clone()),
        ]
    }

    fn get_shader_infos(&self) -> Vec<ShaderInfo> {
        self.shaders.clone()
    }
}

pub struct TwoDPositionSimpleRenderableObject {
    pub vertices: Vec<OnlyTwoDPositionVertex>,
    pub indices: Vec<u32>,
    pub shaders: Vec<ShaderInfo>,
    pub descriptor_set_layout: Option<DescriptorSetLayout>,
}

impl GraphicsObject<OnlyTwoDPositionVertex> for TwoDPositionSimpleRenderableObject {
    fn get_vertices(&self) -> Vec<OnlyTwoDPositionVertex> {
        self.vertices.clone()
    }

    fn get_indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    fn get_resources(&self) -> Vec<(u32, Arc<RwLock<(dyn GraphicsResource + 'static)>>)> {
        vec![]
    }

    fn get_shader_infos(&self) -> Vec<ShaderInfo> {
        self.shaders.clone()
    }
}
