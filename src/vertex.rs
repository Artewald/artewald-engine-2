use ash::vk;
use memoffset::offset_of;
use nalgebra_glm as glm;
use std::{f32::consts::PI, hash::{Hash, Hasher}, num};

use crate::{pipeline_manager::Vertex, vk_allocator::Serializable};

pub const TEST_RECTANGLE: [SimpleVertex; 4] = [
    SimpleVertex::new(glm::Vec3::new(-0.5, -0.5, 0.0), glm::Vec3::new(0.0, 0.0, 1.0), glm::Vec2::new(0.0, 0.0)),
    SimpleVertex::new(glm::Vec3::new(0.5, -0.5, 0.0), glm::Vec3::new(0.0, 1.0, 0.0), glm::Vec2::new(1.0, 0.0)),
    SimpleVertex::new(glm::Vec3::new(0.5, 0.5, 0.0), glm::Vec3::new(1.0, 0.0, 0.0), glm::Vec2::new(1.0, 1.0)),
    SimpleVertex::new(glm::Vec3::new(-0.5, 0.5, 0.0), glm::Vec3::new(1.0, 1.0, 1.0), glm::Vec2::new(0.0, 1.0)),
];

pub const TEST_RECTANGLE_INDICES: [u32; 6] = [
    0, 1, 2,
    2, 3, 0,
];

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct SimpleVertex {
    pub position: glm::Vec3,
    pub color: glm::Vec3,
    pub tex_coord: glm::Vec2,
}

impl SimpleVertex {
    pub const fn new(position: glm::Vec3, color: glm::Vec3, tex_coord: glm::Vec2) -> Self {
        Self {
            position,
            color,
            tex_coord,
        }
    }
}

impl Vertex for SimpleVertex {
    fn get_input_binding_description(&self) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<SimpleVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    fn get_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
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

impl Hash for SimpleVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position.iter().for_each(|&i| i.to_bits().hash(state));
        self.color.iter().for_each(|&i| i.to_bits().hash(state));
        self.tex_coord.iter().for_each(|&i| i.to_bits().hash(state));
    }
}

impl PartialEq for SimpleVertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position &&
        self.color == other.color &&
        self.tex_coord == other.tex_coord
    }
}

impl Eq for SimpleVertex {}

impl Serializable for SimpleVertex {
    fn to_u8(&self) -> Vec<u8> {
        let vertex_bytes: [u8; std::mem::size_of::<Self>()] = unsafe { std::mem::transmute(*self) };
        vertex_bytes.to_vec()
    }
}

impl Serializable for u32 {
    fn to_u8(&self) -> Vec<u8> {
        let index_bytes: [u8; std::mem::size_of::<Self>()] = unsafe { std::mem::transmute(*self) };
        index_bytes.to_vec()
    }
}

// ========================================================================================================================================

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct OnlyTwoDPositionVertex {
    pub position: glm::Vec2,
    pub _padding: f32,
}

impl Vertex for OnlyTwoDPositionVertex {
    fn get_input_binding_description(&self) -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    fn get_attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
        let position_attribute_description = vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: offset_of!(Self, position) as u32,
        };

        vec![position_attribute_description]
    }
}

impl Hash for OnlyTwoDPositionVertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position.iter().for_each(|&i| i.to_bits().hash(state));
    }
}

impl PartialEq for OnlyTwoDPositionVertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
    }
}

impl Eq for OnlyTwoDPositionVertex {}

impl Serializable for OnlyTwoDPositionVertex {
    fn to_u8(&self) -> Vec<u8> {
        let vertex_bytes: [u8; std::mem::size_of::<Self>()] = unsafe { std::mem::transmute(*self) };
        vertex_bytes.to_vec()
    }
}


pub fn generate_circle_type_one(radius: f32, num_points: usize) -> (Vec<OnlyTwoDPositionVertex>, Vec<u32>) {
    let points = calculate_circle_points(radius, num_points);
    let mut vertices = vec![OnlyTwoDPositionVertex { position: glm::Vec2::new(0.0, 0.0), _padding: 0.0}];
    let mut indices = Vec::new();
    
    for point in points {
        vertices.push(OnlyTwoDPositionVertex { position: point, _padding: 0.0});
    }

    for i in 0..(num_points - 1) {
        indices.push(0);
        indices.push((i + 1) as u32);
        indices.push((i + 2) as u32);
    }

    indices.push(0);
    indices.push(num_points as u32);
    indices.push(1);

    indices.reverse();
    (vertices, indices)
}

pub fn generate_circle_type_two(radius: f32, num_points: usize) -> (Vec<OnlyTwoDPositionVertex>, Vec<u32>) {
    // if num_points % 2 != 0 {
    //     panic!("Number of points must be even for this type of circle");
    // }
    
    let points = calculate_circle_points(radius, num_points);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    for point in points {
        vertices.push(OnlyTwoDPositionVertex { position: point, _padding: 0.0});
    }

    indices.push(0);
    indices.push(num_points as u32 - 1);
    indices.push(1);

    // For 12 points the following would be the middle indices:
    for i in 1..(num_points as u32 / 2 - 1) {
        indices.push(i);
        indices.push(num_points as u32 - i);
        indices.push(num_points as u32 - i - 1);
        indices.push(i);
        indices.push(num_points as u32 - i - 1);
        indices.push(i + 1);
    }

    indices.push(num_points as u32 / 2);
    indices.push(num_points as u32 / 2 - 1);
    indices.push(num_points as u32 / 2 + 1);

    println!("{:?}", indices.len());

    // indices.reverse();
    (vertices, indices)
}

pub fn generate_circle_type_three(radius: f32, num_points: usize) -> (Vec<OnlyTwoDPositionVertex>, Vec<u32>) {
    let points = calculate_circle_points(radius, num_points);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    for point in points {
        vertices.push(OnlyTwoDPositionVertex { position: point, _padding: 0.0});
    }

    for i in (0..(num_points as u32 - 2)).step_by(2) {
        indices.push(i);
        indices.push(i + 2);
        indices.push(i + 1);
    }
    indices.push(num_points as u32 - 2);
    indices.push(0);
    indices.push(num_points as u32 - 1);

    let num_levels = (num_points as f64 / 4.0).ceil() as u32 - 1;
    for i in 1..=num_levels {
        let step_by = i * 2;
        let num_triangles_in_level = (num_points as f64 / (3.0 * i as f64)).ceil() as usize - 1;
        let mut index = 0;
        for _ in 0..num_triangles_in_level as u32 {
            indices.push(index % num_points as u32);
            indices.push((index + step_by * 2) % num_points as u32);
            indices.push((index + step_by) % num_points as u32);
            index += step_by * 2;
        }
    }


    (vertices, indices)
}

fn calculate_circle_points(radius: f32, num_points: usize) -> Vec<glm::Vec2> {
    (0..num_points).map(|i| {
        let angle = i as f32 * 2.0 * PI / num_points as f32;
        glm::Vec2::new(radius * angle.cos(), radius * angle.sin())
    }).collect()
}