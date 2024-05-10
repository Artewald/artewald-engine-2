

// The manager needs (per shader):
// - all object types, and how many of each type to draw
// - all vertices (for all objects)
//     - also which indexes in the vec that belongs to which type of object
// - all indices
//     - also which indexes in the vec that belongs to which type of object
// - all static textures (if the shader is set up to have multiple textures per object we need to have multiple texture arrays)
// - all uniform buffer data that is the same for the object type (not dynamic)
// - all "uniform buffer" data (if the shader is set up to have multiple uniform buffers per object we need to have multiple shader storage buffers)
//     - This is the per instance object data (UNIFORM_BUFFER_DYNAMIC)

use std::{any::Any, borrow::Cow, collections::HashMap};

use ash::{vk::{self, DescriptorSet, DescriptorSetLayoutBinding, DescriptorType, PhysicalDevice, Queue, Sampler, StructureType}, Device, Instance};
use image::GenericImageView;
use rand::distributions::uniform;

use crate::{free_allocations_add_error_string, graphics_objects::{Renderable, ResourceID}, pipeline_manager::{GraphicsResourceType, PipelineConfig}, sampler_manager::{self, SamplerConfig, SamplerManager}, vk_allocator::{AllocationInfo, VkAllocator}, vk_controller::{ObjectID, VerticesIndicesHash, VkController}};

type ObjectType = VerticesIndicesHash;
type NumInstances = u32;

pub struct ObjectManager {
    pub data_used_in_shader: HashMap<PipelineConfig, DataUsedInShader>,
}

pub struct DataUsedInShader {
    objects: HashMap<ObjectID, Box<dyn Renderable>>,
    object_type_num_instances: HashMap<ObjectType, NumInstances>,
    object_type_vertices_bytes_indices: HashMap<ObjectType, (u32, u32)>,
    object_type_indices_indices: HashMap<ObjectType, (u32, u32)>,
    // TODO: object_type_textures_dynamic_bytes_indices: HashMap<ObjectID, (u32, u32)>,
    object_id_uniform_buffer_dynamic_bytes_indices: HashMap<(ObjectID, ResourceID), (u32, u32)>,
    vertices: Vec<u32>,
    indices: Vec<u32>,
    textures: HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>,
    uniform_buffer_update_callback: HashMap<(ObjectType, ResourceID), fn() -> Vec<u8>>,
    // TODO: textures_dynamic: Vec<u32>,
    uniform_buffers: HashMap<(ObjectType, ResourceID), AllocationInfo>,
    uniform_buffer_dynamic: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>,
    descriptor_sets: Vec<DescriptorSet>,
    descriptor_type_data: HashMap<ObjectType, Vec<(ResourceID, DescriptorType, DescriptorSetLayoutBinding)>>,
}

impl DataUsedInShader {
    pub fn new(device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, graphics_queue: &Queue, object: (ObjectID, Box<dyn Renderable>), static_resource_callback: Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>, sampler_manager: &mut SamplerManager, allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let object_type = object.1.get_vertices_and_indices_hash();
        let vertices_data = object.1.get_vertex_byte_data();
        let indices_data = object.1.get_indices();
        let mut textures = HashMap::new();
        let mut uniform_buffers = HashMap::new();
        let mut uniform_buffer_dynamic = HashMap::new();
        let mut object_type_uniform_buffer_dynamic_bytes_indices = HashMap::new();
        
        let mut descriptor_set_layout_data = Vec::new();
        for (resource_id, resource_callback, layout_binding) in static_resource_callback {
            match resource_callback() {
                GraphicsResourceType::Texture(image) => {
                    let mut allocation = match allocator.create_device_local_image(image, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1, false) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            let mut allocations = Vec::new();
                            for (allocation, _) in textures.values() {
                                allocations.push(allocation);
                            }
                            for allocation in uniform_buffers.values() {
                                allocations.push(allocation);
                            }
                            free_allocations_add_error_string!(allocator, allocations, error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    let mip_levels = allocation.get_mip_levels().unwrap();
                    // The format needs to be the same as the format read in [`VkAllocator::create_device_local_image`]
                    match allocator.create_image_view(&mut allocation, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR, mip_levels) {
                        Ok(_) => (),
                        Err(e) => {
                            let mut error_str = e.to_string();
                            let mut allocations = Vec::new();
                            allocations.push(&allocation);
                            for (allocation, _) in textures.values() {
                                allocations.push(allocation);
                            }
                            for allocation in uniform_buffers.values() {
                                allocations.push(allocation);
                            }
                            free_allocations_add_error_string!(allocator, allocations, error_str);
                            return Err(Cow::from(error_str));
                        },
                    }
                    
                    let sampler_config = SamplerConfig {
                        s_type: StructureType::SAMPLER_CREATE_INFO,
                        mag_filter: vk::Filter::LINEAR,
                        min_filter: vk::Filter::LINEAR,
                        address_mode_u: vk::SamplerAddressMode::REPEAT,
                        address_mode_v: vk::SamplerAddressMode::REPEAT,
                        address_mode_w: vk::SamplerAddressMode::REPEAT,
                        anisotropy_enable: vk::TRUE,
                        border_color: vk::BorderColor::INT_OPAQUE_BLACK,
                        unnormalized_coordinates: vk::FALSE,
                        compare_enable: vk::FALSE,
                        compare_op: vk::CompareOp::ALWAYS,
                        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                        mip_lod_bias: 0.0,
                        min_lod: 0.0,
                        max_lod: allocation.get_mip_levels().unwrap() as f32,
                    };
                    let sampler = sampler_manager.get_or_create_sampler(device, instance, physical_device, sampler_config, allocator)?;
                    textures.insert((object_type, resource_id), (allocation, sampler));
                    descriptor_set_layout_data.push((resource_id, DescriptorType::COMBINED_IMAGE_SAMPLER, layout_binding));
                },
                GraphicsResourceType::UniformBuffer(buffer) => {
                    let allocation = match allocator.create_uniform_buffers(buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            let mut allocations = Vec::new();
                            for (allocation, _) in textures.values() {
                                allocations.push(allocation);
                            }
                            for allocation in uniform_buffers.values() {
                                allocations.push(allocation);
                            }
                            free_allocations_add_error_string!(allocator, allocations, error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    uniform_buffers.insert((object_type, resource_id), allocation);
                    descriptor_set_layout_data.push((resource_id, DescriptorType::UNIFORM_BUFFER, layout_binding));
                },
                x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
            }
        }

        for (resource_id, resource) in object.1.get_object_resources() {
            let resource_lock = resource.read().unwrap();
            match resource_lock.get_resource() {
                GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                    let allocation = match allocator.create_uniform_buffers(buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            let mut allocations = Vec::new();
                            for (allocation, _) in textures.values() {
                                allocations.push(allocation);
                            }
                            for allocation in uniform_buffers.values() {
                                allocations.push(allocation);
                            }
                            for (allocation, _) in uniform_buffer_dynamic.values() {
                                allocations.push(allocation);
                            }
                            free_allocations_add_error_string!(allocator, allocations, error_str);
                            return Err(Cow::from(error_str));
                        },
                    };

                    uniform_buffer_dynamic.insert((object_type, resource_id), (allocation, buffer));
                    // object_id_uniform_buffer_dynamic_bytes_indices: HashMap<(ObjectID, ResourceID), (u32, u32)>,
                    object_type_uniform_buffer_dynamic_bytes_indices.insert((object.0, resource_id), (0, buffer.len() as u32 - 1));
                    descriptor_set_layout_data.push((resource_id, DescriptorType::UNIFORM_BUFFER_DYNAMIC, resource_lock.get_descriptor_set_layout_binding()));
                },
                x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
            }
        }

        let mut objects = HashMap::new();
        objects.insert(object.0, object.1);
        
        let mut object_type_num_instances = HashMap::new();
        object_type_num_instances.insert(object_type, 1);


        Ok(Self {
            objects,
            object_type_num_instances: todo!(),
            object_type_vertices_bytes_indices: todo!(),
            object_type_indices_indices: todo!(),
            object_id_uniform_buffer_dynamic_bytes_indices: object_type_uniform_buffer_dynamic_bytes_indices,
            vertices: todo!(),
            indices: todo!(),
            textures,
            uniform_buffer_update_callback: todo!(),
            uniform_buffers,
            uniform_buffer_dynamic,
            descriptor_sets: todo!(),
            descriptor_type_data: todo!(),
        })
    }

    
}
