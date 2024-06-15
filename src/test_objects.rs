use std::{collections::{hash_map, HashMap}, hash::{self, Hash, Hasher}, sync::{Arc, RwLock}};
use ash::{vk::{self, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, StructureType}, Device};
use image::DynamicImage;
use nalgebra_glm as glm;

use crate::{graphics_objects::{GraphicsObject, ResourceID, TextureResource, UniformBufferResource}, pipeline_manager::{ObjectInstanceGraphicsResource, ObjectInstanceGraphicsResourceType, ObjectTypeGraphicsResource, ObjectTypeGraphicsResourceType, ShaderInfo}, vertex::{OnlyTwoDPositionVertex, SimpleVertex}, vk_allocator::{Serializable, VkAllocator}, vk_controller::VerticesIndicesHash};

// =========================================== Resources ===========================================

// #[derive(Debug, Clone, Copy, Default)]
// #[repr(C, align(16))]
// pub struct ViewProjectionObject {
//     // pub view: glm::Mat4,
//     // pub proj: glm::Mat4,
//     pub vp: glm::Mat4,
// }

// impl Serializable for ViewProjectionObject {
//     fn to_u8(&self) -> Vec<u8> {
//         // let view = self.view.as_slice();
//         // let proj = self.proj.as_slice();
//         let vp = self.vp.as_slice();
//         let mut result = Vec::with_capacity(std::mem::size_of::<ViewProjectionObject>());
//         // for i in 0..16 {
//         //     result.extend_from_slice(&view[i].to_ne_bytes());
//         // }
//         // for i in 0..16 {
//         //     result.extend_from_slice(&proj[i].to_ne_bytes());
//         // }
//         for i in 0..16 {
//             result.extend_from_slice(&vp[i].to_ne_bytes());
//         }

//         result
//     }
// }

// pub struct ModelMatrixObject {
//     pub model_matrix: glm::Mat4,
// }

// impl Serializable for ModelMatrixObject {
//     fn to_u8(&self) -> Vec<u8> {
//         let model_matrix = self.model_matrix.as_slice();
//         let mut result = Vec::with_capacity(std::mem::size_of::<ModelMatrixObject>());
//         for i in 0..16 {
//             result.extend_from_slice(&model_matrix[i].to_ne_bytes());
//         }

//         result
//     }
// }

impl Serializable for glm::Mat4 {
    fn to_u8(&self) -> Vec<u8> {
        let mat = self.as_slice();
        let mut result = Vec::with_capacity(std::mem::size_of::<glm::Mat4>());
        for i in 0..16 {
            result.extend_from_slice(&mat[i].to_ne_bytes());
        }

        result
    }
}

// =========================================== Objects ===========================================

pub struct SimpleRenderableObject {
    pub vertices: Vec<SimpleVertex>,
    pub indices: Vec<u32>,
    pub model_matrix: Arc<RwLock<UniformBufferResource<glm::Mat4>>>,
    pub shaders: Vec<ShaderInfo>,
    // pub descriptor_set_layout: Option<DescriptorSetLayout>,
    pub view_projection: Arc<RwLock<UniformBufferResource<glm::Mat4>>>,
    pub texture: Arc<RwLock<TextureResource>>,
}       

impl GraphicsObject<SimpleVertex> for SimpleRenderableObject {
    fn get_vertices(&self) -> Vec<SimpleVertex> {
        self.vertices.clone()
    }

    fn get_indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    fn get_instance_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectInstanceGraphicsResource + 'static)>>)> {
        vec![
            (ResourceID(1), self.model_matrix.clone()),
        ]
    }

    fn get_shader_infos(&self) -> Vec<ShaderInfo> {
        self.shaders.clone()
    }
    
    fn get_vertices_and_indices_hash(&self) -> VerticesIndicesHash {
        let mut hasher = hash::DefaultHasher::new();
        self.vertices.iter().for_each(|vertex| vertex.hash(&mut hasher));
        self.indices.iter().for_each(|index| index.hash(&mut hasher));
        VerticesIndicesHash(hasher.finish())
    }
    
    fn get_type_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectTypeGraphicsResource + 'static)>>)> {
        vec![
            (ResourceID(2), self.view_projection.clone()),
            (ResourceID(3), self.texture.clone()),
        ]
    }
    
}

pub struct TwoDPositionSimpleRenderableObject {
    pub vertices: Vec<OnlyTwoDPositionVertex>,
    pub indices: Vec<u32>,
    pub shaders: Vec<ShaderInfo>,
}

impl GraphicsObject<OnlyTwoDPositionVertex> for TwoDPositionSimpleRenderableObject {
    fn get_vertices(&self) -> Vec<OnlyTwoDPositionVertex> {
        self.vertices.clone()
    }

    fn get_indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    fn get_instance_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectInstanceGraphicsResource + 'static)>>)> {
        vec![]
    }

    fn get_shader_infos(&self) -> Vec<ShaderInfo> {
        self.shaders.clone()
    }
    
    fn get_vertices_and_indices_hash(&self) -> VerticesIndicesHash {
        let mut hasher = hash::DefaultHasher::new();
        self.vertices.iter().for_each(|vertex| vertex.hash(&mut hasher));
        self.indices.iter().for_each(|index| index.hash(&mut hasher));
        VerticesIndicesHash(hasher.finish())
    }
    
    fn get_type_resources(&self) -> Vec<(ResourceID, Arc<RwLock<(dyn ObjectTypeGraphicsResource + 'static)>>)> {
        vec![]
    }
    
    
}
