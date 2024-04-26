use std::{borrow::Cow, collections::{hash_map, HashMap}, fmt::Formatter, path::PathBuf, sync::Arc, time::Instant};

use ash::{vk::{self, CommandPool, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPool, DescriptorSet, DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType, Queue, Sampler, StructureType, WriteDescriptorSet}, Device};
use image::DynamicImage;
use nalgebra_glm as glm;

use crate::{pipeline_manager::{GraphicsResource, GraphicsResourceType, PipelineConfig, ShaderInfo, Vertex}, vertex::SimpleVertex, vk_allocator::{AllocationInfo, Serializable, VkAllocator}, vk_controller::{self, VkController}};

macro_rules! free_allocations_add_error_string {
    ($allocator: expr, $allocations: expr, $error_string: expr) => {
        for allocation in $allocations {
            let error = $allocator.free_memory_allocation(allocation);
            if let Err(err) = error {
                $error_string.push_str(&format!("\n{}", err));
            }
        }
    };
}

type ResourceAllocation = (ResourceID, vk::DescriptorSetLayoutBinding, AllocationInfo, DescriptorType, Option<Sampler>);
pub type ResourceID = u32;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub proj: glm::Mat4,
}

impl Serializable for UniformBufferObject {
    fn to_u8(&self) -> Vec<u8> {
        let model = self.model.as_slice();
        let view = self.view.as_slice();
        let proj = self.proj.as_slice();
        let mut result = Vec::with_capacity(std::mem::size_of::<UniformBufferObject>());
        for i in 0..16 {
            result.extend_from_slice(&model[i].to_ne_bytes());
        }
        for i in 0..16 {
            result.extend_from_slice(&view[i].to_ne_bytes());
        }
        for i in 0..16 {
            result.extend_from_slice(&proj[i].to_ne_bytes());
        }

        result
    }
}

#[derive(Clone)]
pub struct UniformBufferResource<T: Clone> {
    pub buffer: T,
    pub binding: u32,
}

impl<T: Clone + Serializable> GraphicsResource for UniformBufferResource<T> {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> crate::pipeline_manager::GraphicsResourceType {
        GraphicsResourceType::UniformBuffer(self.buffer.to_u8())
    }
}

pub struct TextureResource {
    pub image: DynamicImage,
    pub binding: u32,
    pub stage: vk::ShaderStageFlags,
    pub sampler: Sampler,
}

impl GraphicsResource for TextureResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: self.stage,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> GraphicsResourceType {
        GraphicsResourceType::Texture((self.image.clone(), self.sampler.clone()))
    }
}

#[derive(Clone)]
pub struct SimpleObjectTextureResource {
    pub path: PathBuf,
    pub binding: u32,
    pub sampler: Sampler,
}

impl GraphicsResource for SimpleObjectTextureResource {
    fn get_descriptor_set_layout_binding(&self) -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding {
            binding: self.binding,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            p_immutable_samplers: std::ptr::null(),
        }
    }

    fn get_resource(&self) -> GraphicsResourceType {
        GraphicsResourceType::Texture((image::open(self.path.clone()).unwrap(), self.sampler.clone()))
    }
}

pub trait GraphicsObject<T: Vertex> {
    fn get_vertices(&self) -> Vec<T>;
    fn get_indices(&self) -> Vec<u32>;
    fn get_resources(&self) -> Vec<(ResourceID, Arc<dyn GraphicsResource>)>;
    fn get_shader_infos(&self) -> Vec<ShaderInfo>;
}

pub trait Renderable {
    fn take_vertex_allocation(&mut self) -> AllocationInfo;
    fn take_index_allocation(&mut self) -> AllocationInfo;
    fn take_extra_resource_allocations(&mut self) -> Vec<ResourceAllocation>;
    fn borrow_vertex_allocation(&self) -> Option<&AllocationInfo>;
    fn borrow_index_allocation(&self) -> Option<&AllocationInfo>;
    fn get_num_indecies(&self) -> usize;
    fn borrow_extra_resource_allocations(&self) -> Vec<(u32, vk::DescriptorSetLayoutBinding, &AllocationInfo, DescriptorType, Option<Sampler>)>;
    fn get_pipeline_config(&self) -> PipelineConfig;
    fn borrow_descriptor_sets(&self) -> &[DescriptorSet];
    fn cleanup(&mut self, device: &Device, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>>;
    fn update_extra_resource_allocations(&mut self, device: &Device, start_time: Instant, frame_index: usize, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>>;
}

pub struct ObjectToRender<T: Vertex> {
    vertex_allocation: Option<AllocationInfo>,
    index_allocation: Option<AllocationInfo>,
    extra_resource_allocations: Vec<ResourceAllocation>,
    pipeline_config: PipelineConfig,
    original_object: Arc<dyn GraphicsObject<T>>,
    descriptor_sets: Vec<DescriptorSet>,
}


impl<T: Vertex + Clone + 'static> ObjectToRender<T> {
    pub fn new(device: &Device, original_object: Arc<dyn GraphicsObject<T>>, swapchain_format: vk::Format, depth_format: vk::Format, command_pool: &CommandPool, graphics_queue: &Queue, msaa_samples: vk::SampleCountFlags, descriptor_pool: &DescriptorPool, mip_levels: u32, allocator: &mut VkAllocator) -> Result<Self, Cow<'static, str>> {
        let vertices = original_object.get_vertices();
        let vertex_data = vertices.iter().map(|v| v.to_u8()).flatten().collect::<Vec<u8>>();
        let vertex_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &vertex_data, vk::BufferUsageFlags::VERTEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => return Err(Cow::from(e)),
        };
        let indices = original_object.get_indices();
        let index_data = indices.iter().map(|i| i.to_ne_bytes()).flatten().collect::<Vec<u8>>();
        let index_allocation = match allocator.create_device_local_buffer(command_pool, graphics_queue, &index_data, vk::BufferUsageFlags::INDEX_BUFFER, false) {
            Ok(alloc) => alloc,
            Err(e) => {
                let mut error_str = e.to_string();
                free_allocations_add_error_string!(allocator, vec![vertex_allocation], error_str);
                return Err(Cow::from(error_str));
            },
        
        };
        let mut descriptor_set_layout_bindings: Vec<DescriptorSetLayoutBinding> = Vec::with_capacity(original_object.get_resources().len());
        let mut extra_resource_allocations: Vec<ResourceAllocation> = Vec::with_capacity(original_object.get_resources().len());
        for (id, resource) in original_object.get_resources() {
            match resource.get_resource() {
                GraphicsResourceType::UniformBuffer(buffer) => {
                    let allocation = match allocator.create_uniform_buffers(buffer.len(), VkController::MAX_FRAMES_IN_FLIGHT) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            free_allocations_add_error_string!(allocator, vec![vertex_allocation, index_allocation], error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    extra_resource_allocations.push((id, resource.get_descriptor_set_layout_binding(), allocation, DescriptorType::UNIFORM_BUFFER, None));
                    descriptor_set_layout_bindings.push(resource.get_descriptor_set_layout_binding());
                }
                GraphicsResourceType::Texture((image, sampler)) => {
                    let mut allocation = match allocator.create_device_local_image(image, command_pool, graphics_queue, u32::MAX, vk::SampleCountFlags::TYPE_1, false) {
                        Ok(alloc) => alloc,
                        Err(e) => {
                            let mut error_str = e.to_string();
                            free_allocations_add_error_string!(allocator, vec![vertex_allocation, index_allocation], error_str);
                            return Err(Cow::from(error_str));
                        },
                    };
                    // The format needs to be the same as the format read in [`VkAllocator::create_device_local_image`]
                    match allocator.create_image_view(&mut allocation, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR, mip_levels) {
                        Ok(_) => (),
                        Err(e) => {
                            let mut error_str = e.to_string();
                            free_allocations_add_error_string!(allocator, vec![vertex_allocation, index_allocation, allocation], error_str);
                            return Err(Cow::from(error_str));
                        },
                    }

                    extra_resource_allocations.push((id, resource.get_descriptor_set_layout_binding(), allocation, DescriptorType::COMBINED_IMAGE_SAMPLER, Some(sampler)));
                    descriptor_set_layout_bindings.push(resource.get_descriptor_set_layout_binding());
                }
            }
        }

        let vertex_sample = match vertices.first() {
            Some(v) => v.clone(),
            None => return Err("No vertices found when trying to create graphics object for rendering".into()),
        };

        let pipeline_config = PipelineConfig::new(
            device, 
            original_object.get_shader_infos(),
            vertex_sample.get_input_binding_description(),
            vertex_sample.get_attribute_descriptions(),
            &descriptor_set_layout_bindings,
            msaa_samples,
            swapchain_format,
            depth_format,
            allocator,
        )?;

        let descriptor_sets = Self::create_descriptor_set(device, descriptor_pool, pipeline_config.borrow_descriptor_set_layout(), &extra_resource_allocations, vk_controller::VkController::MAX_FRAMES_IN_FLIGHT as u32, allocator);

        Ok(Self {
            vertex_allocation: Some(vertex_allocation),
            index_allocation: Some(index_allocation),
            extra_resource_allocations,
            pipeline_config,
            original_object,
            descriptor_sets,
        })

        //Err("Not implemented".into())
    }

    pub fn get_pipeline_config(&self) -> PipelineConfig {
        self.pipeline_config.clone()
    }

    fn create_descriptor_set(device: &Device, descriptor_pool: &DescriptorPool, descriptor_set_layout: &DescriptorSetLayout, resource_allocations: &[ResourceAllocation], frames_in_flight: u32, allocator: &mut VkAllocator) -> Vec<DescriptorSet> {
        let layouts = vec![*descriptor_set_layout; frames_in_flight as usize];
        let alloc_info = DescriptorSetAllocateInfo {
            s_type: StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            descriptor_pool: *descriptor_pool,
            descriptor_set_count: frames_in_flight,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&alloc_info).unwrap()
        };

        for i in 0..frames_in_flight {
            let mut descriptor_writes: Vec<WriteDescriptorSet> = Vec::with_capacity(resource_allocations.len());
            
            for (_, descriptor_set_layout_binding, allocation_info, descriptor_type, sampler_option) in resource_allocations.iter() {
                let write_descriptor = match *descriptor_type {
                    DescriptorType::UNIFORM_BUFFER => {
                        // println!("Offset: {}", unsafe{allocation_info.get_uniform_pointers()[i as usize].offset_from(allocation_info.get_uniform_pointers()[0])});
                        let offset = unsafe {allocation_info.get_uniform_pointers()[i as usize].offset_from(allocation_info.get_uniform_pointers()[0])} as u64;
                        let buffer_info = DescriptorBufferInfo {
                            buffer: allocation_info.get_buffer().unwrap(),
                            offset,
                            range: std::mem::size_of::<UniformBufferObject>() as u64,
                        };
                        vk::WriteDescriptorSet {
                            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                            dst_set: descriptor_sets[i as usize],
                            dst_binding: descriptor_set_layout_binding.binding,
                            dst_array_element: 0,
                            descriptor_type: DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            p_buffer_info: &buffer_info,
                            p_image_info: std::ptr::null(),
                            p_texel_buffer_view: std::ptr::null(),
                            ..Default::default()
                        }
                    },
                    DescriptorType::COMBINED_IMAGE_SAMPLER => {
                        let image_info = DescriptorImageInfo {
                            sampler: sampler_option.as_ref().unwrap().clone(),
                            image_view: allocation_info.get_image_view().unwrap(),
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        };
                        
                        vk::WriteDescriptorSet {
                            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                            dst_set: descriptor_sets[i as usize],
                            dst_binding: descriptor_set_layout_binding.binding,
                            dst_array_element: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            p_image_info: &image_info,
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

        descriptor_sets
    }
}

impl<T: Vertex> Renderable for ObjectToRender<T> {
    fn take_vertex_allocation(&mut self) -> AllocationInfo {
        self.vertex_allocation.take().unwrap()
    }

    fn take_index_allocation(&mut self) -> AllocationInfo {
        self.index_allocation.take().unwrap()
    }

    fn borrow_extra_resource_allocations(&self) -> Vec<(u32, DescriptorSetLayoutBinding, &AllocationInfo, DescriptorType, Option<Sampler>)> {
        self.extra_resource_allocations.iter().map(|(id, binding, alloc, descriptor_type, sampler)| (*id, *binding, alloc, *descriptor_type, *sampler)).collect()
    }

    fn get_pipeline_config(&self) -> PipelineConfig {
        self.pipeline_config.clone()
    }
    
    fn get_num_indecies(&self) -> usize {
        self.original_object.get_indices().len()
    }
    
    fn borrow_vertex_allocation(&self) -> Option<&AllocationInfo> {
        self.vertex_allocation.as_ref()
    }
    
    fn borrow_index_allocation(&self) -> Option<&AllocationInfo> {
        self.index_allocation.as_ref()
    }
    
    fn take_extra_resource_allocations(&mut self) -> Vec<ResourceAllocation> {
        self.extra_resource_allocations.drain(..).collect()
    }
    
    fn borrow_descriptor_sets(&self) -> &[DescriptorSet] {
        &self.descriptor_sets
    }
    
    fn cleanup(&mut self, device: &Device, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        // vertex_allocation: Option<AllocationInfo>,
        // index_allocation: Option<AllocationInfo>,
        // extra_resource_allocations: Vec<ResourceAllocation>,
        // pipeline_config: PipelineConfig,
        // original_object: Arc<dyn GraphicsObject<T>>,
        // descriptor_sets: Vec<DescriptorSet>,

        allocator.free_memory_allocation(self.take_vertex_allocation())?;
        allocator.free_memory_allocation(self.take_index_allocation())?;
        for (_, _, allocation, _, _) in self.take_extra_resource_allocations() {
            allocator.free_memory_allocation(allocation)?;
        }

        self.descriptor_sets.clear();
        Ok(())
    }
    
    fn update_extra_resource_allocations(&mut self, device: &Device, start_time: Instant, frame_index: usize, allocator: &mut VkAllocator) -> Result<(), Cow<'static, str>> {
        let resources = self.original_object.get_resources();
        
        for (id, _, allocation_info, descriptor_type, _) in self.borrow_extra_resource_allocations() {
            match descriptor_type {
                DescriptorType::UNIFORM_BUFFER => {
                    // let elapsed = start_time.elapsed().as_secs_f32();
                    // let mut ubo = UniformBufferObject {
                    //     model: glm::rotate(&glm::identity(), elapsed * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0)),
                    //     view: glm::look_at(&glm::vec3(2.0, 2.0, 2.0), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 0.0, 1.0)),
                    //     proj: glm::perspective(1.7777, 90.0_f32.to_radians(), 0.1, 10.0),
                    // };
                    // ubo.proj[(1, 1)] *= -1.0;
                    let data = resources.iter().find(|(r_id, _)| id == *r_id).unwrap();
                    let binary_data = match data.1.get_resource() {
                        GraphicsResourceType::UniformBuffer(buffer) => buffer,
                        _ => {
                            return Err("Not implemented resource type for uniform buffer".into());
                        },
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(binary_data.as_ptr() as *const std::ffi::c_void, allocation_info.get_uniform_pointers()[frame_index], binary_data.len());
                    }
                },
                DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    ()
                },
                _ => {
                    panic!("Not implemented for descriptor type {:?}", descriptor_type.as_raw());
                },
            };
        }

        Ok(())
    }
    
    
}