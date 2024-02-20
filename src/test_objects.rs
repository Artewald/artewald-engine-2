use std::{collections::{hash_map, HashMap}, sync::Arc};
use image::DynamicImage;
use nalgebra_glm as glm;

use crate::{graphics_objects::{GraphicsObject, TextureResource, UniformBufferObject, UniformBufferResource}, vertex::SimpleVertex, vk_allocator::Serializable};

pub struct SimpleRenderableObject {
    pub vertices: Vec<SimpleVertex>,
    pub indices: Vec<u32>,
    pub uniform_buffer: Arc<UniformBufferResource<UniformBufferObject>>,
    pub texture: Arc<TextureResource>,
    
}

impl GraphicsObject<SimpleVertex> for SimpleRenderableObject {
    fn get_vertices(&self) -> Vec<SimpleVertex> {
        self.vertices.clone()
    }

    fn get_indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    fn get_resources(&self) -> Vec<std::sync::Arc<dyn crate::pipeline_manager::GraphicsResource>> {
        vec![
            self.uniform_buffer.clone(),
            self.texture.clone(),
        ]
    }

    fn get_shader_infos(&self) -> Vec<crate::pipeline_manager::ShaderInfo> {
        todo!()
    }

    fn get_msaa_samples(&self) -> ash::vk::SampleCountFlags {
        todo!()
    }
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