

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

use std::{borrow::Cow, collections::{HashMap, HashSet}, hash::Hash};

use ash::{vk::{self, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, PhysicalDevice, Queue, Sampler, StructureType, WriteDescriptorSet}, Device, Instance};
use image::DynamicImage;

use crate::{free_allocations_add_error_string, graphics_objects::{Renderable, ResourceID}, pipeline_manager::{GraphicsResourceType, PipelineConfig}, sampler_manager::{SamplerConfig, SamplerManager}, vk_allocator::{AllocationInfo, VkAllocator}, vk_controller::{ObjectID, VerticesIndicesHash, VkController}};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct NumInstances(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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

    pub fn new(objects_to_add: Vec<(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, descriptor_pool: &DescriptorPool, graphics_queue: &Queue, sampler_manager: &mut SamplerManager, current_frame: u32, allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let mut textures = HashMap::new();
        let mut uniform_buffers = HashMap::new();
        let mut dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)> = HashMap::new();
        let mut object_id_uniform_buffer_dynamic_bytes_indices = HashMap::new();
        let mut object_type_vertices_bytes_indices = HashMap::new();
        let mut object_type_indices_bytes_indices = HashMap::new();
        let mut object_global_resource_update_callback = HashMap::new();
        let mut descriptor_type_data = Vec::new();
        let mut object_types = HashSet::new();
        let mut objects = HashMap::new();
        let pipeline_config = objects_to_add[0].1.get_pipeline_config();
        let mut vertices_data = Vec::new();
        let mut indices_data = Vec::new();

        let (object_type_data, object_type_num_instances) = Self::get_object_type_data_and_num_instances(&objects_to_add);

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
            for (resource_id, resource) in objects_to_add.iter().find(|obj| obj.1.get_vertices_and_indices_hash() == object_type.0).unwrap().1.get_object_resources() {
                let resource_lock = resource.read().unwrap();
                match resource_lock.get_resource() {
                    GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                        match Self::create_dynamic_uniform_buffer(*object_type, resource_id, *num_instances, buffer.clone(), &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, allocator) {
                            Ok(_) => (),
                            Err(e) => return Err(e),
                        }
    
                        if !descriptor_type_data.iter().any(|x| x.0 == resource_id) {
                            descriptor_type_data.push((resource_id, DescriptorType::UNIFORM_BUFFER_DYNAMIC, resource_lock.get_descriptor_set_layout_binding()));
                        }

                        Self::add_object_vertices_and_indices_if_new_object_type(*object_type, &object_type_data, &mut object_type_vertices_bytes_indices, &mut object_type_indices_bytes_indices, &mut vertices_data, &mut indices_data).unwrap();
                        
                        Self::add_static_resource_callbacks_to_object_global_resource_update_callback_if_new_object_type(*object_type, &object_type_data, &mut object_global_resource_update_callback);
                    },
                    x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
                }
            } 
        }
        
        for object in objects_to_add {
            let object_type = ObjectType(object.1.get_vertices_and_indices_hash());
            let newly_added_object_type = object_types.insert(object_type);
            
            if newly_added_object_type {
                for (resource_id, resource_callback, layout_binding) in object.2 {
                    match resource_callback() {
                        GraphicsResourceType::Texture(image) => {
                            match Self::create_and_add_static_texture(object_type, resource_id, image, device, instance, physical_device, command_pool, graphics_queue, &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, sampler_manager, allocator) {
                                Ok(_) => (),
                                Err(e) => return Err(e),
                            }
                        },
                    GraphicsResourceType::UniformBuffer(buffer) => {
                        match Self::create_and_add_static_uniform_buffer(object_type, resource_id, &buffer, current_frame, &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, allocator) {
                            Ok(_) => (),
                            Err(e) => return Err(e),
                        }
                    },
                    x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
                }
                }
            }
            
            objects.insert(object.0, object.1);
        }
        
        Self::create_dynamic_uniform_buffer_byte_indices(&objects_to_add, &mut object_id_uniform_buffer_dynamic_bytes_indices);
        
        Self::update_and_copy_dynamic_buffer_data_to_gpu(&mut objects, &dynamic_uniform_buffers, &object_id_uniform_buffer_dynamic_bytes_indices);

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

        let descriptor_sets = Self::create_descriptor_sets(device, descriptor_pool, pipeline_config.borrow_descriptor_set_layout().unwrap(), &object_types, &descriptor_type_data, uniform_buffers, textures, dynamic_uniform_buffers, VkController::MAX_FRAMES_IN_FLIGHT as u32, allocator);

        Ok(Self {
            objects,
            object_type_num_instances,
            object_type_vertices_bytes_indices,
            object_type_indices_bytes_indices,
            object_id_uniform_buffer_dynamic_bytes_indices,
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

    pub fn add_objects(&mut self, objects_to_add: Vec<(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, descriptor_pool: &DescriptorPool, graphics_queue: &Queue, sampler_manager: &mut SamplerManager, current_frame: u32, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        let mut textures = HashMap::new();
        let mut uniform_buffers = HashMap::new();
        let mut dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)> = HashMap::new();
        let mut object_id_uniform_buffer_dynamic_bytes_indices = HashMap::new();
        let mut object_type_vertices_bytes_indices = self.object_type_vertices_bytes_indices.clone();
        let mut object_type_indices_bytes_indices = self.object_type_indices_bytes_indices.clone();
        let mut object_global_resource_update_callback = self.object_global_resource_update_callback.clone();
        let descriptor_type_data = self.descriptor_type_data.clone();
        let mut object_types = HashSet::new();
        let mut new_objects = HashMap::new();
        let pipeline_config = objects_to_add[0].1.get_pipeline_config();
        let mut vertices_data = self.vertices.1.clone();
        let mut indices_data = self.indices.1.clone();

        let (object_type_data, mut object_type_num_instances) = Self::get_object_type_data_and_num_instances(&objects_to_add);

        object_type_num_instances.iter_mut().for_each(|(object_type, num_instances)| {
            *num_instances = self.object_type_num_instances.get(object_type).unwrap_or(&NumInstances(0)).clone();
        });

        for (object_type, num_instances) in object_type_num_instances.iter() {
            for (resource_id, resource) in objects_to_add.iter().find(|obj| obj.1.get_vertices_and_indices_hash() == object_type.0).unwrap().1.get_object_resources() {
                let resource_lock = resource.read().unwrap();
                match resource_lock.get_resource() {
                    GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                        match Self::create_dynamic_uniform_buffer(*object_type, resource_id, *num_instances, buffer.clone(), &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, allocator) {
                            Ok(_) => (),
                            Err(e) => return Err(e),
                        }

                        Self::add_object_vertices_and_indices_if_new_object_type(*object_type, &object_type_data, &mut object_type_vertices_bytes_indices, &mut object_type_indices_bytes_indices, &mut vertices_data, &mut indices_data).unwrap();
                        
                        Self::add_static_resource_callbacks_to_object_global_resource_update_callback_if_new_object_type(*object_type, &object_type_data, &mut object_global_resource_update_callback);
                    },
                    x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
                }
            } 
        }
        
        for object in objects_to_add {
            let object_type = ObjectType(object.1.get_vertices_and_indices_hash());
            let newly_added_object_type = object_types.insert(object_type) && !self.object_type_num_instances.contains_key(&object_type);
            
            // TODO: add the ability to override static object type data
            if newly_added_object_type {
                for (resource_id, resource_callback, layout_binding) in object.2 {
                    match resource_callback() {
                        GraphicsResourceType::Texture(image) => {
                            match Self::create_and_add_static_texture(object_type, resource_id, image, device, instance, physical_device, command_pool, graphics_queue, &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, sampler_manager, allocator) {
                                Ok(_) => (),
                                Err(e) => return Err(e),
                            }
                        },
                    GraphicsResourceType::UniformBuffer(buffer) => {
                        match Self::create_and_add_static_uniform_buffer(object_type, resource_id, &buffer, current_frame, &mut textures, &mut uniform_buffers, &mut dynamic_uniform_buffers, allocator) {
                            Ok(_) => (),
                            Err(e) => return Err(e),
                        }
                    },
                    x => eprintln!("You cannot attach resource type that is not static to a specific object type (instance definition), use static resources instead. Currently only textures and uniform buffers (non-dynamic) are supported."),
                }
                }
            }
            
            new_objects.insert(object.0, object.1);
        }
        
        Self::create_dynamic_uniform_buffer_byte_indices(&objects_to_add, &mut object_id_uniform_buffer_dynamic_bytes_indices);
        
        Self::update_and_copy_dynamic_buffer_data_to_gpu(&mut new_objects, &dynamic_uniform_buffers, &object_id_uniform_buffer_dynamic_bytes_indices);

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

        let descriptor_sets = Self::create_descriptor_sets(device, descriptor_pool, pipeline_config.borrow_descriptor_set_layout().unwrap(), &object_types, &descriptor_type_data, uniform_buffers, textures, dynamic_uniform_buffers, VkController::MAX_FRAMES_IN_FLIGHT as u32, allocator);

        todo!("Now we need to update self's data with the new data, but only once the previous allocations are no longer used.");

        Ok(())
    }

    fn create_descriptor_sets(device: &Device, descriptor_pool: &DescriptorPool, descriptor_set_layout: &DescriptorSetLayout, object_types: &HashSet<ObjectType>, descriptor_type_data: &[(ResourceID, DescriptorType, DescriptorSetLayoutBinding)], uniform_buffers: HashMap<(ObjectType, ResourceID), AllocationInfo>, textures: HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, dynamic_uniform_buffers: HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, frames_in_flight: u32, allocator: &mut VkAllocator) -> HashMap<ObjectType, Vec<DescriptorSet>> {
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
                let num_resources = descriptor_type_data.len();
                let mut descriptor_writes: Vec<WriteDescriptorSet> = Vec::with_capacity(num_resources);
                
                // We need this so that the buffer/image info is not dropped before the write descriptor is used
                let mut buffer_infos = Vec::with_capacity(num_resources);
                let mut image_infos = Vec::with_capacity(num_resources);
    
                for (resource_id, descriptor_type, layout_binding) in descriptor_type_data {
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
            descriptor_sets.insert(*object_type, descriptor_sets_local);
        }

        descriptor_sets
    }

    fn get_object_type_data_and_num_instances(objects_to_add: &[(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)]) -> (HashMap<VerticesIndicesHash, (Vec<u8>, Vec<u32>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, HashMap<ObjectType, NumInstances>) {
        let mut object_type_data = HashMap::new();
        let mut object_type_num_instances = HashMap::new();
        objects_to_add.iter().for_each(|(_, object, callbacks)| {
            let object_type = object.get_vertices_and_indices_hash();
            let e = object_type_num_instances.entry(ObjectType(object_type)).or_insert(NumInstances(0));
            e.0 += 1;
            if object_type_data.contains_key(&object_type) {
                return;
            }
            object_type_data.insert(object_type, (object.get_vertex_byte_data(), object.get_indices(), callbacks.clone()));
        });
        (object_type_data, object_type_num_instances)
    }

    fn create_dynamic_uniform_buffer(object_type: ObjectType, resource_id: ResourceID, num_instances: NumInstances, buffer: Vec<u8>, new_textures: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, new_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), AllocationInfo>, new_dynamic_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        let allocation = match allocator.create_uniform_buffers(num_instances.0 as usize * buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                let mut allocations = Vec::new();
                Self::add_hashmap_allocations_to_free(new_textures, new_uniform_buffers, new_dynamic_uniform_buffers, &mut allocations, allocator);
                free_allocations_add_error_string!(allocator, allocations, error_str);
                return Err(Cow::from(error_str));
            },
        };

        new_dynamic_uniform_buffers.insert((object_type, resource_id), (allocation, Vec::new()));
        Ok(())
    }

    fn add_object_vertices_and_indices_if_new_object_type(object_type: ObjectType, object_type_data: &HashMap<VerticesIndicesHash, (Vec<u8>, Vec<u32>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, object_type_vertices_bytes_indices: &mut HashMap<ObjectType, (u32, u32)>, object_type_indices_bytes_indices: &mut HashMap<ObjectType, (u32, u32)>, vertices_data: &mut Vec<u8>, indices_data: &mut Vec<u8>) -> Result<(), Cow<'static, str>> {
        if !object_type_vertices_bytes_indices.contains_key(&object_type) {
            let (object_vertices_data, object_indices, resource_callbacks) = object_type_data.get(&object_type.0).unwrap();
            let object_indices_data = object_indices.iter().map(|x| x.to_ne_bytes()).flatten().collect::<Vec<u8>>();
            object_type_vertices_bytes_indices.insert(object_type, (vertices_data.len() as u32, (vertices_data.len() + object_vertices_data.len()) as u32 - 1));
            vertices_data.extend_from_slice(object_vertices_data);
            object_type_indices_bytes_indices.insert(object_type, (indices_data.len() as u32, (indices_data.len() + object_indices.len()) as u32 - 1));    
            indices_data.extend_from_slice(&object_indices_data);
        }
        Ok(())
    }

    fn add_static_resource_callbacks_to_object_global_resource_update_callback_if_new_object_type(object_type: ObjectType, object_type_data: &HashMap<VerticesIndicesHash, (Vec<u8>, Vec<u32>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)>, object_global_resource_update_callback: &mut HashMap<(ObjectType, ResourceID), fn() -> GraphicsResourceType>) {
        let resource_callbacks = object_type_data.get(&object_type.0).unwrap().2;
        for (resource_id, resource_callback, _) in resource_callbacks.iter() {
            if object_global_resource_update_callback.contains_key(&(object_type, *resource_id)) {
                continue;
            }
            object_global_resource_update_callback.insert((object_type, *resource_id), *resource_callback);
        }
    }

    fn create_and_add_static_texture(object_type: ObjectType, resource_id: ResourceID, image: DynamicImage, device: &Device, instance: &Instance, physical_device: &PhysicalDevice, command_pool: &vk::CommandPool, graphics_queue: &Queue, new_textures: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, new_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), AllocationInfo>, new_dynamic_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, sampler_manager: &mut SamplerManager, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        let mut allocation = match allocator.create_device_local_image(image, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                let mut allocations = Vec::new();
                Self::add_hashmap_allocations_to_free(new_textures, new_uniform_buffers, new_dynamic_uniform_buffers, &mut allocations, allocator);
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
                allocations.push(allocation);
                Self::add_hashmap_allocations_to_free(new_textures, new_uniform_buffers, new_dynamic_uniform_buffers, &mut allocations, allocator);
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
        Ok(())
    }

    fn create_and_add_static_uniform_buffer(object_type: ObjectType, resource_id: ResourceID, buffer: &[u8], current_frame: u32, new_textures: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, new_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), AllocationInfo>, new_dynamic_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        let allocation = match allocator.create_uniform_buffers(buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                let mut allocations = Vec::new();
                Self::add_hashmap_allocations_to_free(new_textures, new_uniform_buffers, new_dynamic_uniform_buffers, &mut allocations, allocator);
                free_allocations_add_error_string!(allocator, allocations, error_str);
                return Err(Cow::from(error_str));
            },
        };

        unsafe {
            std::ptr::copy_nonoverlapping(buffer.as_ptr() as *const std::ffi::c_void, allocation.get_uniform_pointers()[current_frame as usize], buffer.len());
        }

        new_uniform_buffers.insert((object_type, resource_id), allocation);
        Ok(())
    }

    fn create_dynamic_uniform_buffer_byte_indices(objects_to_add: &[(ObjectID, Box<dyn Renderable>, Vec<(ResourceID, fn() -> GraphicsResourceType, DescriptorSetLayoutBinding)>)], object_id_uniform_buffer_dynamic_bytes_indices: &mut HashMap<(ObjectID, ResourceID), (u32, u32)>) {
        let mut number_of_allocated_dynamic_uniform_buffers_per_object_and_resource_id = HashMap::new();
        objects_to_add.iter().for_each(|(object_id, object, _)| {
            let object_type = ObjectType(object.get_vertices_and_indices_hash());
            for (resource_id, resource) in object.get_object_resources() {
                let resource_lock = resource.read().unwrap();
                match resource_lock.get_resource() {
                    GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                        let current_resource_allocation_number = number_of_allocated_dynamic_uniform_buffers_per_object_and_resource_id.entry((*object_id, resource_id)).or_insert(0);
                        object_id_uniform_buffer_dynamic_bytes_indices.insert((*object_id, resource_id), ((*current_resource_allocation_number as usize *buffer.len()) as u32, ((*current_resource_allocation_number + 1) as usize * buffer.len()) as u32 - 1));
                        *current_resource_allocation_number += 1;
                    },
                    x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. If you want to use the same buffer for all objects of this type, use the static resource callbacks. Currently only dynamic uniform buffers are supported."),
                }
            }
        });
    }

    fn update_and_copy_dynamic_buffer_data_to_gpu(objects: &mut HashMap<ObjectID, Box<dyn Renderable>>, dynamic_uniform_buffers: &HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, object_id_uniform_buffer_dynamic_bytes_indices: &HashMap<(ObjectID, ResourceID), (u32, u32)>) {
        objects.iter().for_each(|(object_id, object)| {
            let object_type = ObjectType(object.get_vertices_and_indices_hash());
            for (resource_id, resource) in object.get_object_resources() {
                let resource_lock = resource.read().unwrap();
                match resource_lock.get_resource() {
                    GraphicsResourceType::DynamicUniformBuffer(buffer) => {
                        let (allocation_info, buffer) = dynamic_uniform_buffers.get(&(object_type, resource_id)).expect("Dynamic uniform buffer not found for object type. This should never happen. Was the dynamic uniform buffer added to the object type?");
                        let (start, end) = object_id_uniform_buffer_dynamic_bytes_indices.get(&(*object_id, resource_id)).expect("Dynamic uniform buffer bytes indices not found for object id. This should never happen. Was the dynamic uniform buffer added to the object id?");
                        if buffer.len() != (*end - *start + 1) as usize {
                            eprintln!("The dynamic uniform buffer size does not match the size of the buffer that was allocated for it. This should never happen.");
                        }
                        buffer[(*start as usize)..(*end as usize)].copy_from_slice(&buffer[(*start as usize)..(*end as usize + 1)]);
                    },
                    x => eprintln!("You cannot attach resource type that is not dynamic/bindless to a specific object type (instance definition), use dynamic buffers instead. Currently only dynamic uniform buffers are supported."),
                }
            }
        });

        dynamic_uniform_buffers.iter().for_each(|(_, (allocation_info, buffer))| {
            unsafe {
                std::ptr::copy_nonoverlapping(buffer.as_ptr() as *const std::ffi::c_void, allocation_info.get_uniform_pointers()[0], buffer.len());
            }
        });
    }

    fn add_hashmap_allocations_to_free(textures: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Sampler)>, uniform_buffers: &mut HashMap<(ObjectType, ResourceID), AllocationInfo>, dynamic_uniform_buffers: &mut HashMap<(ObjectType, ResourceID), (AllocationInfo, Vec<u8>)>, allocations: &mut Vec<AllocationInfo>, allocator: &mut VkAllocator) {
        for (_, (allocation, _)) in textures.drain() {
            allocations.push(allocation);
        }
        for (_, allocation) in uniform_buffers.drain() {
            allocations.push(allocation);
        }
        for (_, (allocation, _)) in dynamic_uniform_buffers.drain() {
            allocations.push(allocation);
        }
    }
}
