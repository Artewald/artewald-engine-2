use std::{fmt::Formatter, path::PathBuf};

use ash::vk;
use nalgebra_glm as glm;

use crate::{vertex::Vertex, vk_allocator::Serializable};

#[derive(Debug, Clone, Copy)]
#[repr(C, align(16))]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}

fn get_shader_stage_flag_names() -> Vec<&'static str> {
    Vec::new()
}

#[derive(Clone)]
pub struct ShaderInfo {
    path: PathBuf,
    stage: vk::ShaderStageFlags,
    entry_point: String,
}

impl ShaderInfo {
    pub fn new(path: PathBuf, stage: vk::ShaderStageFlags, entry_point: String) -> Self {
        if !path.exists() {
            panic!("Shader file {:?} does not exist", path);
        }
        ShaderInfo { path, stage, entry_point }
    }
}

impl std::fmt::Debug for ShaderInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShaderInfo {{ path: {:?}, stage: {}, entry_point: {} }}", self.path, self.stage.as_raw(), self.entry_point)
    }
}

#[derive(Debug, Clone)]
pub enum DescriptorContent<T: Serializable + std::fmt::Debug> {
    UniformBuffer(T),
    Texture(PathBuf),
}

#[derive(Clone)]
pub struct RenderableObject<T: Serializable + std::fmt::Debug, Y: Serializable + std::fmt::Debug> {
    shaders: Vec<ShaderInfo>,
    vertex_binding_info: vk::VertexInputBindingDescription,
    vertex_attribute_info: Vec<vk::VertexInputAttributeDescription>,
    vertices: Vec<T>,
    binding_descriptions: Vec<(DescriptorContent<Y>, vk::DescriptorSetLayoutBinding)>,
}

impl<T: Serializable + std::fmt::Debug, Y: Serializable + std::fmt::Debug> RenderableObject<T, Y> {
    pub fn new(
        shaders: Vec<ShaderInfo>,
        vertex_binding_info: vk::VertexInputBindingDescription,
        vertex_attribute_info: Vec<vk::VertexInputAttributeDescription>,
        vertices: Vec<T>,
        binding_descriptions: Vec<(DescriptorContent<Y>, vk::DescriptorSetLayoutBinding)>,
    ) -> Self {
        if shaders.len() < 2 {
            panic!("RenderableObject must have at least 2 shaders");
        }
        if vertex_attribute_info.len() == 0 {
            panic!("RenderableObject must have at least 1 vertex attribute");
        }
        if vertices.len() == 1 {
            panic!("RenderableObject must have at least 3 vertex");
        }
        RenderableObject {
            shaders,
            vertex_binding_info,
            vertex_attribute_info,
            vertices,
            binding_descriptions,
        }
    }

    pub fn get_shaders(&self) -> &Vec<ShaderInfo> {
        &self.shaders
    }

    pub fn get_vertex_binding_info(&self) -> &vk::VertexInputBindingDescription {
        &self.vertex_binding_info
    }

    pub fn get_vertex_attribute_info(&self) -> &Vec<vk::VertexInputAttributeDescription> {
        &self.vertex_attribute_info
    }

    pub fn get_vertices(&self) -> &Vec<T> {
        &self.vertices
    }

    pub fn get_binding_descriptions(&self) -> &Vec<(DescriptorContent<Y>, vk::DescriptorSetLayoutBinding)> {
        &self.binding_descriptions
    }
}

impl<T: Serializable + std::fmt::Debug, Y: Serializable + std::fmt::Debug> std::fmt::Debug for RenderableObject<T, Y>   {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "RenderableObject {{ shaders: {:?}, vertices: {:?}, ", self.shaders, self.vertices)?;
        write!(f, "vertex_binding_info {{ binding: {}, stride: {}, input_rate: {} }}, ", self.vertex_binding_info.binding, self.vertex_binding_info.stride, self.vertex_binding_info.input_rate.as_raw())?; //, vertex_binding_info: {:?},
        for attribute_info in self.vertex_attribute_info.iter() {
            write!(f, "vertex_attribute_info {{ location: {}, binding: {}, format: {}, offset: {} }}, ", attribute_info.location, attribute_info.binding, attribute_info.format.as_raw(), attribute_info.offset)?;
        }; //vertex_attribute_info: {:?}
        for (descriptor_content, descriptor_set_layout_binding) in self.binding_descriptions.iter() {
            write!(f, "descriptor_content: {:?}, descriptor_set_layout_binding {{ binding: {}, descriptor_type: {}, descriptor_count: {}, stage_flags: {}, p_immutable_samplers: {:?} }}, ", descriptor_content, descriptor_set_layout_binding.binding, descriptor_set_layout_binding.descriptor_type.as_raw(), descriptor_set_layout_binding.descriptor_count, descriptor_set_layout_binding.stage_flags.as_raw(), descriptor_set_layout_binding.p_immutable_samplers)?; //, binding_descriptions: {:?} }}
        }
        Ok(())
    }
}