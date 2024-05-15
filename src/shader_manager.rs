

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

use std::{any::Any, borrow::Cow, collections::{HashMap, HashSet}};

use ash::{vk::{self, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, PhysicalDevice, Queue, Sampler, StructureType, WriteDescriptorSet}, Device, Instance};
use image::GenericImageView;
use rand::distributions::uniform;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{free_allocations_add_error_string, graphics_objects::{Renderable, ResourceID}, pipeline_manager::{GraphicsResourceType, PipelineConfig}, sampler_manager::{self, SamplerConfig, SamplerManager}, vk_allocator::{AllocationInfo, VkAllocator}, vk_controller::{ObjectID, VerticesIndicesHash, VkController}};

// type ObjectType = VerticesIndicesHash;
struct NumInstances(pub u32);

pub struct ObjectType(VerticesIndicesHash);

pub struct ObjectManager {
    pub data_used_in_shader: HashMap<PipelineConfig, DataUsedInShader>,
}

pub struct DataUsedInShader {
    objects: HashMap<ObjectID, Box<dyn Renderable>>,
    object_type_num_instances: HashMap<ObjectType, NumInstances>,
    object_type_vertices_bytes_indices: HashMap<ObjectType, (u32, u32)>,
    object_type_indices_bytes_indices: HashMap<ObjectType, (u32, u32)>,
    // TODO: object_type_textures_dynamic_bytes_indices: HashMap<ObjectID, (u32, u32)>,
    object_id_uniform_buffer_dynamic_bytes_indices: HashMap<(ObjectID, ResourceID), (u32, u32)>,
    vertices: (AllocationInfo, Vec<u8>),
    indices: (AllocationInfo, Vec<u8>),
    textures: HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>,
    object_global_resource_update_callback: HashMap<(ObjectType, ResourceID), fn() -> GraphicsResourceType>,
    // TODO: textures_dynamic: Vec<u32>,
    uniform_buffers: HashMap<(ObjectType, ResourceID), AllocationInfo>,
    dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>,
    descriptor_type_data: Vec<(ResourceID, DescriptorType, DescriptorSetLayoutBinding)>,
    descriptor_sets: HashMap<ObjectType, Vec<DescriptorSet>>,
}

impl DataUsedInShader {
    
    pub fn new(objects_to_add: Vec<(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, descriptor_pool: &DescriptorPool, graphics_queue: &Queue, sampler_manager: &mut SamplerManager, allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let mut textures = HashMap::new();
        let mut uniform_buffers = HashMap::new();
        let mut dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)> = HashMap::new();
        let mut object_type_uniform_buffer_dynamic_bytes_indices = HashMap::new();
        let mut object_type_vertices_bytes_indices = HashMap::new();
        let mut object_type_indices_bytes_indices = HashMap::new();
        let mut object_global_resource_update_callback = HashMap::new();
        let mut descriptor_type_data: Vec<(ResourceID, DescriptorType, DescriptorSetLayoutBinding)> = Vec::new();
        let mut object_types = HashSet::new();
        let mut objects = HashMap::new();
        let mut object_type_num_instances = HashMap::new();
        let pipeline_config = objects_to_add[0].1.get_pipeline_config();
        let mut vertices_data = Vec::new();
        let mut indices_data = Vec::new();

        objects_to_add.iter().for_each(|(_, object, _)| {
            let e = object_type_num_instances.entry(object.get_vertices_and_indices_hash()).or_insert(0u32);
            *e += 1;
        });

        for (resource_id, resource_callback, layout_binding) in objects_to_add.first().unwrap().2.iter() {
            match resource_callback() {
                GraphicsResourceType::Texture(image) => {
                    descriptor_type_data.push((*resource_id, DescriptorType::COMBINED_IMAGE_SAMPLER, *layout_binding));
                },
                GraphicsResourceType::UniformBuffer(buffer) => {
                    descriptor_type_data.push((*resource_id, DescriptorType::UNIFORM_BUFFER, *layout_binding));
                },
                x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
            }
        }

        for (object_type, num_instances) in object_type_num_instances.iter() {
            for (resource_id, resource) in objects_to_add.first().unwrap().1.get_object_resources() {
                let resource_lock = resource.read().unwrap();
                match resource_lock.get_resource() {
                    GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                        let allocation = match allocator.create_uniform_buffers(*num_instances as usize * buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
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
                                for (allocation, _) in dynamic_uniform_buffers.values() {
                                    allocations.push(allocation);
                                }
                                free_allocations_add_error_string!(allocator, allocations, error_str);
                                return Err(Cow::from(error_str));
                            },
                        };
    
                        dynamic_uniform_buffers.insert(resource_id, (allocation, buffer));
    
                        object_type_uniform_buffer_dynamic_bytes_indices.insert((object.0, resource_id), (0, buffer.len() as u32 - 1));
                        descriptor_type_data.push(((*object_type, resource_id), DescriptorType::UNIFORM_BUFFER_DYNAMIC, resource_lock.get_descriptor_set_layout_binding()));
                        
                        let object_vertices_data = object.1.get_vertex_byte_data();
                        let object_indices_data = object.1.get_indices().iter().map(|x| x.to_ne_bytes()).flatten().collect::<Vec<u8>>();
                
                        vertices_data.extend_from_slice(&object_vertices_data);
                        indices_data.extend_from_slice(&object_indices_data);
        
                        object_type_vertices_bytes_indices.insert(object_type, (0, object_vertices_data.len() as u32 - 1));
                        object_type_indices_bytes_indices.insert(object_type, (0, object_indices_data.len() as u32 - 1));    
                        
                        for (resource_id, resource_callback, _) in object.2 {
                            object_global_resource_update_callback.insert((object_type, resource_id), resource_callback);
                        }
                    },
                    x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
                }
            } 
        }

        for object in objects_to_add {
            let object_type = object.1.get_vertices_and_indices_hash();
            let newly_added_object_type = object_types.insert(object_type);
    
            if newly_added_object_type {
                for (resource_id, resource_callback, layout_binding) in object.2 {
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
                        },
                        x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
                    }
                }
            }

            objects.insert(object.0, object.1);
        }

        let vertex_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &vertices_data, vk::BufferUsageFlags::VERTEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => return Err(Cow::from(e)),
        };
        let index_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &indices_data, vk::BufferUsageFlags::INDEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                free_allocations_add_error_string!(allocator, vec![vertex_allocation], error_str);
                return Err(Cow::from(error_str));
            },
        };

        let descriptor_sets = Self::create_descriptor_sets(device, descriptor_pool, pipeline_config.borrow_descriptor_set_layout().unwrap(), &object_types, descriptor_type_data, uniform_buffers, textures, dynamic_uniform_buffers, VkController::MAX_FRAMES_IN_FLIGHT as u32, allocator);

        Ok(Self {
            objects,
            object_type_num_instances,
            object_type_vertices_bytes_indices,
            object_type_indices_bytes_indices,
            object_id_uniform_buffer_dynamic_bytes_indices: object_type_uniform_buffer_dynamic_bytes_indices,
            vertices: (vertex_allocation, vertices_data),
            indices: (index_allocation, indices_data),
            textures,
            object_global_resource_update_callback,
            uniform_buffers,
            dynamic_uniform_buffers,
            descriptor_type_data,
            descriptor_sets,
        })
    }

    fn create_descriptor_sets(device: &Device, descriptor_pool: &DescriptorPool, descriptor_set_layout: &DescriptorSetLayout, object_types: &HashSet<ObjectType>, descriptor_type_data: HashMap<ObjectType, Vec<(ResourceID, DescriptorType, DescriptorSetLayoutBinding)>>, uniform_buffers: HashMap<(ObjectType, ResourceID), AllocationInfo>, textures: HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, frames_in_flight: u32, allocator: &mut VkAllocator) -> HashMap<ObjectType, Vec<DescriptorSet>> {
        let mut descriptor_sets = HashMap::new();

        for object_type in object_types {
            let layouts = vec![*descriptor_set_layout; frames_in_flight as usize];
            let alloc_info = DescriptorSetAllocateInfo {
                s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptor_pool: *descriptor_pool,
                descriptor_set_count: frames_in_flight,
                p_set_layouts: layouts.as_ptr(),
                ..Default::default()
            };
    
            let descriptor_sets_local = unsafe {
                device.allocate_descriptor_sets(&alloc_info).unwrap()
            };
    
            for i in 0..frames_in_flight {
                let num_resources = descriptor_type_data.get(object_type).unwrap().len();
                let mut descriptor_writes: Vec<WriteDescriptorSet> = Vec::with_capacity(num_resources);
                
                // We need this so that the buffer/image info is not dropped before the write descriptor is used
                let mut buffer_infos = Vec::with_capacity(num_resources);
                let mut image_infos = Vec::with_capacity(num_resources);
    
                for (resource_id, descriptor_type, layout_binding) in descriptor_type_data.get(object_type).unwrap() {
                    let write_descriptor = match *descriptor_type {
                        DescriptorType::UNIFORM_BUFFER => {
                            let allocation_info = uniform_buffers.get(&(*object_type, *resource_id)).expect("Uniform buffer not found for object type. This should never happen. Was the uniform buffer added to the object type?");
                            let offset = unsafe {allocation_info.get_uniform_pointers()[i as usize].offset_from(allocation_info.get_uniform_pointers()[0])} as u64;
                            let size = (allocation_info.get_memory_end()-allocation_info.get_memory_start())/allocation_info.get_uniform_pointers().len().max(1) as u64;
                            // println!("Offset: {}, size: {}", offset , size);
                            let buffer = allocation_info.get_buffer().unwrap();
                            let buffer_info = DescriptorBufferInfo {
                                buffer,
                                offset,
                                range: size,
                            };
    
                            buffer_infos.push(buffer_info);
                            let buffer_info = buffer_infos.last().unwrap();
                            vk::WriteDescriptorSet {
                                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                                dst_set: descriptor_sets_local[i as usize],
                                dst_binding: layout_binding.binding,
                                dst_array_element: 0,
                                descriptor_type: DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                                p_buffer_info: buffer_info,
                                p_image_info: std::ptr::null(),
                                p_texel_buffer_view: std::ptr::null(),
                                ..Default::default()
                            }
                        },
                        DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                            let (allocation_info, _) = dynamic_uniform_buffers.get(&(*object_type, *resource_id)).expect("Dynamic uniform buffer not found for object type. This should never happen. Was the dynamic uniform buffer added to the object type?");
                            let offset = unsafe {allocation_info.get_uniform_pointers()[i as usize].offset_from(allocation_info.get_uniform_pointers()[0])} as u64;
                            let size = (allocation_info.get_memory_end()-allocation_info.get_memory_start())/allocation_info.get_uniform_pointers().len().max(1) as u64;
                            let buffer = allocation_info.get_buffer().unwrap();
                            let buffer_info = DescriptorBufferInfo {
                                buffer,
                                offset,
                                range: size,
                            };
    
                            buffer_infos.push(buffer_info);
                            let buffer_info = buffer_infos.last().unwrap();
                            vk::WriteDescriptorSet {
                                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                                dst_set: descriptor_sets_local[i as usize],
                                dst_binding: layout_binding.binding,
                                dst_array_element: 0,
                                descriptor_type: DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                                descriptor_count: 1,
                                p_buffer_info: buffer_info,
                                p_image_info: std::ptr::null(),
                                p_texel_buffer_view: std::ptr::null(),
                                ..Default::default()
                            }
                        },
                        DescriptorType::COMBINED_IMAGE_SAMPLER => {
                            let (allocation_info, sampler) = textures.get(&(*object_type, *resource_id)).expect("Texture not found for object type. This should never happen. Was the texture added to the object type?");
                            let image_info = DescriptorImageInfo {
                                sampler: sampler.clone(),
                                image_view: allocation_info.get_image_view().unwrap(),
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            };
                            
                            image_infos.push(image_info);
                            let image_info = image_infos.last().unwrap();
    
                            vk::WriteDescriptorSet {
                                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                                dst_set: descriptor_sets_local[i as usize],
                                dst_binding: layout_binding.binding,
                                dst_array_element: 0,
                                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: 1,
                                p_image_info: image_info,
                                p_texel_buffer_view: std::ptr::null(),
                                ..Default::default()
                            }
    
                        },
                        _ => {
                            panic!("Not implemented for descriptor type {:?}", descriptor_type.as_raw());
                        },
                    };
                    descriptor_writes.push(write_descriptor);
                }
    
                unsafe {
                    device.update_descriptor_sets(&descriptor_writes, &vec![]);
                }
            }
        }

        descriptor_sets
    }

    pub fn add_objects(&mut self, objects_to_add: Vec<(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, descriptor_pool: &DescriptorPool, graphics_queue: &Queue, sampler_manager: &mut SamplerManager, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        objects_to_add.iter().for_each(|(_, object, _)| {
            let e = self.object_type_num_instances.entry(object.get_vertices_and_indices_hash()).or_insert(0u32);
            *e += 1;
        });

        let mut new_textures = HashMap::new();
        let mut new_uniform_buffers = HashMap::new();
        

        for object in objects_to_add {
            let object_type = object.1.get_vertices_and_indices_hash();
            let newly_added_object_type = !self.object_type_num_instances.keys().any(|x| x == &object_type);

            if newly_added_object_type {
                let mut descriptor_set_layout_data = Vec::new();
                for (resource_id, resource_callback, layout_binding) in object.2 {
                    match resource_callback() {
                        GraphicsResourceType::Texture(image) => {
                            let mut allocation = match allocator.create_device_local_image(image, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1, false) {
                                Ok(alloc) => alloc,
                                Err(e) => {
                                    let mut error_str = e.to_string();
                                    let mut allocations = Vec::new();
                                    for (allocation, _) in new_textures.values() {
                                        allocations.push(allocation);
                                    }
                                    for allocation in new_uniform_buffers.values() {
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
                                    for (allocation, _) in new_textures.values() {
                                        allocations.push(allocation);
                                    }
                                    for allocation in new_uniform_buffers.values() {
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
                            new_textures.insert((object_type, resource_id), (allocation, sampler));
                            descriptor_set_layout_data.push((resource_id, DescriptorType::COMBINED_IMAGE_SAMPLER, layout_binding));
                        },
                        GraphicsResourceType::UniformBuffer(buffer) => {
                            let allocation = match allocator.create_uniform_buffers(buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
                                Ok(alloc) => alloc,
                                Err(e) => {
                                    let mut error_str = e.to_string();
                                    let mut allocations = Vec::new();
                                    for (allocation, _) in new_textures.values() {
                                        allocations.push(allocation);
                                    }
                                    for allocation in new_uniform_buffers.values() {
                                        allocations.push(allocation);
                                    }
                                    free_allocations_add_error_string!(allocator, allocations, error_str);
                                    return Err(Cow::from(error_str));
                                },
                            };
                            // uniform_buffers.insert((object_type, resource_id), allocation);
                            descriptor_set_layout_data.push((resource_id, DescriptorType::UNIFORM_BUFFER, layout_binding));
                        },
                        x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
                    }
                }

                let num_objects = self.object_type_num_instances.get(&object_type).unwrap(); 
                for (resource_id, resource) in object.1.get_object_resources() {
                    let resource_lock = resource.read().unwrap();
                    match resource_lock.get_resource() {
                        GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                            let allocation = match allocator.create_uniform_buffers(*num_objects as usize * buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
                                Ok(alloc) => alloc,
                                Err(e) => {
                                    let mut error_str = e.to_string();
                                    let mut allocations = Vec::new();
                                    for (allocation, _) in new_textures.values() {
                                        allocations.push(allocation);
                                    }
                                    for allocation in new_uniform_buffers.values() {
                                        allocations.push(allocation);
                                    }
                                    for (allocation, _) in new_dynamic_uniform_buffers.values() {
                                        allocations.push(allocation);
                                    }
                                    free_allocations_add_error_string!(allocator, allocations, error_str);
                                    return Err(Cow::from(error_str));
                                },
                            };
        
                            new_dynamic_uniform_buffers.insert((object_type, resource_id), (allocation, buffer));

                            object_type_uniform_buffer_dynamic_bytes_indices.insert((object.0, resource_id), (0, buffer.len() as u32 - 1));
                            descriptor_set_layout_data.push((resource_id, DescriptorType::UNIFORM_BUFFER_DYNAMIC, resource_lock.get_descriptor_set_layout_binding()));
                            
                            let object_vertices_data = object.1.get_vertex_byte_data();
                            let object_indices_data = object.1.get_indices().iter().map(|x| x.to_ne_bytes()).flatten().collect::<Vec<u8>>();
                    
                            vertices_data.extend_from_slice(&object_vertices_data);
                            indices_data.extend_from_slice(&object_indices_data);
            
                            object_type_vertices_bytes_indices.insert(object_type, (0, object_vertices_data.len() as u32 - 1));
                            object_type_indices_bytes_indices.insert(object_type, (0, object_indices_data.len() as u32 - 1));    
                            
                            for (resource_id, resource_callback, _) in object.2 {
                                object_global_resource_update_callback.insert((object_type, resource_id), resource_callback);
                            }
                        },
                        x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
                    }
                }

                descriptor_type_data.insert(object_type, descriptor_set_layout_data);
            }

            objects.insert(object.0, object.1);
        }

        let vertex_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &vertices_data, vk::BufferUsageFlags::VERTEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => return Err(Cow::from(e)),
        };
        let index_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &indices_data, vk::BufferUsageFlags::INDEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                free_allocations_add_error_string!(allocator, vec![vertex_allocation], error_str);
                return Err(Cow::from(error_str));
            },
        };

        let descriptor_sets = Self::create_descriptor_sets(device, descriptor_pool, pipeline_config.borrow_descriptor_set_layout().unwrap(), &object_types, descriptor_type_data, uniform_buffers, textures, dynamic_uniform_buffers, VkController::MAX_FRAMES_IN_FLIGHT as u32, allocator);

        Ok(())
    }
}
