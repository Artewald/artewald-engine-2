use std::{collections::{hash_map, HashMap}, sync::{Arc, RwLock}};
use ash::{vk::{DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, StructureType}, Device};
use nalgebra_glm as glm;

use crate::{graphics_objects::{GraphicsObject, TextureResource, UniformBufferObject, UniformBufferResource}, pipeline_manager::{GraphicsResource, GraphicsResourceType, ShaderInfo}, vertex::SimpleVertex, vk_allocator::{Serializable, VkAllocator}};

pub struct SimpleRenderableObject {
    pub vertices: Vec<SimpleVertex>,
    pub indices: Vec<u32>,
    pub uniform_buffer: Arc<RwLock<UniformBufferResource<UniformBufferObject>>>,
    pub texture: Arc<RwLock<TextureResource>>,
    pub shaders: Vec<ShaderInfo>,
    pub descriptor_set_layout: Option<DescriptorSetLayout>,
}       

impl SimpleRenderableObject {

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
    
    // fn get_or_create_descriptor_set_layout(&self, device: &Device, allocator: &mut VkAllocator) -> DescriptorSetLayout {
    //     if self.descriptor_set_layout.is_some() {
    //         return self.descriptor_set_layout.unwrap();
    //     }

    //     let bindings: Vec<DescriptorSetLayoutBinding> = self.get_resources().iter().map(|resource| resource.get_descriptor_set_layout_binding()).collect();
    //     let create_info = DescriptorSetLayoutCreateInfo {
    //         flags: DescriptorSetLayoutCreateFlags::empty(),
    //         binding_count: bindings.len() as u32,
    //         p_bindings: bindings.as_ptr(),
    //         ..Default::default()
    //     };

    //     unsafe {
    //         device.create_descriptor_set_layout(&create_info, Some(&allocator.get_allocation_callbacks()))
    //     }.unwrap()
    // }
}


fn load_model(path: &str) -> (Vec<SimpleVertex>, Vec<u32>) {
    let (models, _) = tobj::load_obj(path, &tobj::LoadOptions::default()).unwrap();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut unique_vertices: HashMap<SimpleVertex, u32> = HashMap::new();

    for model in models {
        let mesh = model.mesh;
        for i in 0..mesh.indices.len() {
            let index = mesh.indices[i] as usize;
            let vertex = SimpleVertex {
                position: glm::vec3(mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]),
                color: glm::vec3(1.0, 1.0, 1.0),
                tex_coord: glm::vec2(mesh.texcoords[index * 2], 1.0 - mesh.texcoords[index * 2 + 1]),
            };
    
            if let hash_map::Entry::Vacant(e) = unique_vertices.entry(vertex) {
                e.insert(vertices.len() as u32);
                vertices.push(vertex);
            }
            indices.push(*unique_vertices.get(&vertex).unwrap());
        }
    }

    (vertices, indices)
}