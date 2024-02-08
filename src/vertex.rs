use ash::vk;
use memoffset::offset_of;
use nalgebra_glm as glm;
use std::hash::{Hash, Hasher};

use crate::vk_allocator::Serializable;

pub const TEST_RECTANGLE: [Vertex; 4] = [
    Vertex::new(glm::Vec3::new(-0.5, -0.5, 0.0), glm::Vec3::new(0.0, 0.0, 1.0), glm::Vec2::new(0.0, 0.0)),
    Vertex::new(glm::Vec3::new(0.5, -0.5, 0.0), glm::Vec3::new(0.0, 1.0, 0.0), glm::Vec2::new(1.0, 0.0)),
    Vertex::new(glm::Vec3::new(0.5, 0.5, 0.0), glm::Vec3::new(1.0, 0.0, 0.0), glm::Vec2::new(1.0, 1.0)),
    Vertex::new(glm::Vec3::new(-0.5, 0.5, 0.0), glm::Vec3::new(1.0, 1.0, 1.0), glm::Vec2::new(0.0, 1.0)),
];

pub const TEST_RECTANGLE_INDICES: [u32; 6] = [
    0, 1, 2,
    2, 3, 0,
];

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub position: glm::Vec3,
    pub color: glm::Vec3,
    pub tex_coord: glm::Vec2,
}

impl Vertex {
    pub const fn new(position: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self {
            position,
            color,
            tex_coord,
        }
    }

    pub fn vertex_input_binding_descriptions() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        // If you add any 64 bit types, you need to change the format to R64G64_SFLOAT and increase the location size to 2
        let position_attribute_description = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, position) as u32,
        };

        let color_attribute_description = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, color) as u32,
        };

        let tex_coord_attribute_description = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 2,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Self, tex_coord) as u32,
        };

        vec![position_attribute_description, color_attribute_description, tex_coord_attribute_description]
    }
}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position.iter().for_each(|&i| i.to_bits().hash(state));
        self.color.iter().for_each(|&i| i.to_bits().hash(state));
        self.tex_coord.iter().for_each(|&i| i.to_bits().hash(state));
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position &&
        self.color == other.color &&
        self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Serializable for Vertex {
    fn to_u8(&self) -> Vec<u8> {
        let vertex_bytes: [u8; std::mem::size_of::<Vertex>()] = unsafe { std::mem::transmute(*self) };
        vertex_bytes.to_vec()
    }
}

impl Serializable for u32 {
    fn to_u8(&self) -> Vec<u8> {
        let index_bytes: [u8; std::mem::size_of::<u32>()] = unsafe { std::mem::transmute(*self) };
        index_bytes.to_vec()
    }
}
