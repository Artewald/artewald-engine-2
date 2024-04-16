use std::{borrow::Cow, collections::HashMap, ffi::c_void, rc::Rc, sync::{Arc, Mutex}};

use ash::{vk::{self, DependencyFlags, StructureType, SystemAllocationScope}, Instance, Device};
use image::DynamicImage;

type MemoryTypeIndex = u32;
type MemoryOffset = vk::DeviceSize;
type MemorySizeRange = (vk::DeviceSize, vk::DeviceSize);
type Alignment = usize;

pub trait Serializable {
    fn to_u8(&self) -> Vec<u8>;
}

#[derive(Debug, Clone)]
pub struct AllocationInfo {
    buffer: Option<vk::Buffer>,
    image: Option<vk::Image>,
    mip_levels: Option<u32>,
    image_view: Option<vk::ImageView>,
    memory_index: MemoryTypeIndex,
    memory_start: MemoryOffset,
    memory_end: vk::DeviceSize,
    memory: vk::DeviceMemory,
    uniform_pointers: Vec<*mut c_void>,
}

#[derive(Debug)]
struct HostAllocationPool {
    start_ptr: *mut u8,
    size: usize,
    alignment: Alignment,
    free_allocations: Vec<(usize, usize)>,
}

pub struct VkAllocator {
    device: Rc<Device>,
    physical_device: vk::PhysicalDevice,
    instance: Rc<Instance>,
    device_allocations: HashMap<MemoryTypeIndex, Vec<(vk::DeviceMemory, Vec<MemorySizeRange>)>>,
    host_allocator: Arc<Mutex<VkHostAllocator>>,
}

pub struct VkHostAllocator {
    host_allocations: HashMap<Alignment, Vec<HostAllocationPool>>,
    allocated_host_pointers: HashMap<*mut c_void, (Alignment, usize)>,
}

// Device memory allocation
impl VkAllocator {
    const DEFAULT_DEVICE_MEMORY_ALLOCATION_BYTE_SIZE: vk::DeviceSize = 256_000_000; // 256 MB 

    pub fn new(instance: Rc<Instance>, physical_device: vk::PhysicalDevice, device: Rc<Device>) -> Self {
        Self {
            device,
            physical_device,
            instance,
            device_allocations: HashMap::new(),
            host_allocator: Arc::new(Mutex::new(VkHostAllocator {
                host_allocations: HashMap::new(),
                allocated_host_pointers: HashMap::new(),
            })),
        }
    }

    pub fn create_uniform_buffers(&mut self, buffer_size: usize, num_buffers: usize) -> Result<AllocationInfo, Cow<'static, str>> {
        let buffer_size = (buffer_size * num_buffers) as u64;

        // let mut uniform_buffers = Vec::with_capacity(num_buffers);
        
        let mut allocation_info = self.create_buffer(buffer_size, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, true)?; //Self::create_buffer(instance, physical_device, device, buffer_size as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, allocator);
        let data_ptr = unsafe {
            self.device.map_memory(allocation_info.get_memory(), allocation_info.get_memory_start(), buffer_size, vk::MemoryMapFlags::empty()).unwrap()
        };
        for i in 0..num_buffers {
            let offset = match (i*buffer_size as usize/num_buffers).try_into() {
                Ok(offset) => offset,
                Err(err) => return Err(Cow::from(format!("Failed to create uniform buffers because: {}", err))),
            };
            allocation_info.uniform_pointers.push(unsafe {data_ptr.offset(offset)});
        }

        Ok(allocation_info)
    }

    pub fn create_buffer(&mut self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags, force_own_memory_block: bool) -> Result<AllocationInfo, Cow<'static, str>> {
        let buffer_info = vk::BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            match self.device.create_buffer(&buffer_info, Some(&self.get_allocation_callbacks())) {
                Ok(buffer) => buffer,
                Err(err) => return Err(Cow::from(format!("Failed to create buffer when creating buffer because: {}", err))),
            }
        };
        
        let memory_requirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: memory_requirements.size,
            memory_type_index: self.find_memory_type( memory_requirements.memory_type_bits, properties)?,
            ..Default::default()
        };

        let mut allocation_info = self.get_allocation(alloc_info.memory_type_index, alloc_info.allocation_size, memory_requirements.alignment, force_own_memory_block)?;

        unsafe {
            match self.device.bind_buffer_memory(buffer, allocation_info.memory, allocation_info.memory_start) {
                Ok(_) => {},
                Err(err) => {
                    self.free_memory_allocation(allocation_info)?;
                    return Err(Cow::from(format!("Failed to bind buffer memory when creating buffer because: {}", err)));
                },
            };
        }

        allocation_info.buffer = Some(buffer);

        Ok(allocation_info)
    }

    pub fn create_device_local_buffer(&mut self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, data: &[u8], buffer_usage: vk::BufferUsageFlags, force_own_memory_block: bool) -> Result<AllocationInfo, Cow<'static, str>> {
        // let data_vec = Self::serializable_vec_to_u8_vec(to_serialize);
        // let data = data_vec.as_slice();

        let size = std::mem::size_of_val(data);

        let staging_allocation = self.create_buffer(size as u64, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, force_own_memory_block)?;
        
        unsafe {
            let mapped_memory_ptr = match self.device.map_memory(staging_allocation.memory, staging_allocation.memory_start, size as u64, vk::MemoryMapFlags::empty()) {
                Ok(ptr) => ptr as *mut u8,
                Err(err) => {
                    self.free_memory_allocation(staging_allocation)?;
                    return Err(Cow::from(format!("Failed to map memory when creating device local buffer because: {}", err)));
                },
            };
            let data_ptr = data.as_ptr();
            std::ptr::copy_nonoverlapping(data_ptr, mapped_memory_ptr, size);
            self.device.unmap_memory(staging_allocation.memory);
        }
        
        let device_local_allocation = self.create_buffer(size as u64, buffer_usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, force_own_memory_block)?;

        self.copy_buffer(&staging_allocation, &device_local_allocation, command_pool, graphics_queue)?;

        if self.free_memory_allocation(staging_allocation).is_err() {
            if self.free_memory_allocation(device_local_allocation).is_err() {
                return Err(Cow::from("Failed to free device local buffer allocation after freeing staging buffer allocation failed!"));
            }
            return Err(Cow::from("Failed to free staging buffer allocation!"));
        }

        Ok(device_local_allocation)
    }

    pub fn create_device_local_buffer_test<T: Serializable>(&mut self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, to_serialize: &[T], buffer_usage: vk::BufferUsageFlags, force_own_memory_block: bool) -> Result<AllocationInfo, Cow<'static, str>> {
        let data_vec = Self::serializable_vec_to_u8_vec(to_serialize);
        let data = data_vec.as_slice();

        let size = std::mem::size_of_val(data);

        let staging_allocation = self.create_buffer(size as u64, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, force_own_memory_block)?;
        
        unsafe {
            let mapped_memory_ptr = match self.device.map_memory(staging_allocation.memory, staging_allocation.memory_start, size as u64, vk::MemoryMapFlags::empty()) {
                Ok(ptr) => ptr as *mut u8,
                Err(err) => {
                    self.free_memory_allocation(staging_allocation)?;
                    return Err(Cow::from(format!("Failed to map memory when creating device local buffer because: {}", err)));
                },
            };
            let data_ptr = data.as_ptr();
            std::ptr::copy_nonoverlapping(data_ptr, mapped_memory_ptr, size);
            self.device.unmap_memory(staging_allocation.memory);
        }
        
        let device_local_allocation = self.create_buffer(size as u64, buffer_usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL, force_own_memory_block)?;

        self.copy_buffer(&staging_allocation, &device_local_allocation, command_pool, graphics_queue)?;

        if self.free_memory_allocation(staging_allocation).is_err() {
            if self.free_memory_allocation(device_local_allocation).is_err() {
                return Err(Cow::from("Failed to free device local buffer allocation after freeing staging buffer allocation failed!"));
            }
            return Err(Cow::from("Failed to free staging buffer allocation!"));
        }

        Ok(device_local_allocation)
    }

    pub fn create_image(&mut self, width: u32, height: u32, mip_levels: u32, num_samples: vk::SampleCountFlags, format: vk::Format, tiling: vk::ImageTiling, usage: vk::ImageUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<AllocationInfo, Cow<'static, str>> {
        let image_info = vk::ImageCreateInfo {
            s_type: StructureType::IMAGE_CREATE_INFO,
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            format,
            tiling,
            initial_layout: vk::ImageLayout::UNDEFINED,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            samples: num_samples,
            flags: vk::ImageCreateFlags::empty(),
            ..Default::default()
        };

        let image = unsafe {
            match self.device.create_image(&image_info, Some(&self.get_allocation_callbacks())) {
                Ok(image) => image,
                Err(err) => return Err(Cow::from(format!("Failed to create image when creating image because: {}", err))),
            }
        };

        let mem_requirements = unsafe {
            self.device.get_image_memory_requirements(image)
        };

        let mut image_allocation = self.get_allocation(self.find_memory_type(mem_requirements.memory_type_bits, properties)?, mem_requirements.size, mem_requirements.alignment, false)?;

        image_allocation.image = Some(image);

        unsafe {
            match self.device.bind_image_memory(image, image_allocation.memory, image_allocation.memory_start) {
                Ok(_) => {},
                Err(err) => {
                    self.free_memory_allocation(image_allocation)?;
                    return Err(Cow::from(format!("Failed to bind image memory when creating image because: {}", err)));
                },
            };
        }

        Ok(image_allocation)
    }    

    pub fn create_device_local_image(&mut self, image: DynamicImage, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, max_mip_levels: u32, num_samples: vk::SampleCountFlags, force_own_memory_block: bool) -> Result<AllocationInfo, Cow<'static, str>> {
        // let binding = image::open("./assets/images/viking_room.png").unwrap();
        let image = image.to_rgba8();
        let image_size: vk::DeviceSize = image.dimensions().0 as vk::DeviceSize * image.dimensions().1 as vk::DeviceSize * 4 as vk::DeviceSize;
        
        let mip_levels = (((image.dimensions().0 as f32).max(image.dimensions().1 as f32).log2().floor() + 1.0) as u32).min(max_mip_levels);

        let staging_allocation = self.create_buffer(image_size, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT, force_own_memory_block)?;

        unsafe {
            let data_ptr = match self.device.map_memory(staging_allocation.memory, staging_allocation.memory_start, image_size, vk::MemoryMapFlags::empty()) {
                Ok(ptr) => ptr as *mut u8,
                Err(err) => {
                    self.free_memory_allocation(staging_allocation)?;
                    return Err(Cow::from(format!("Failed to map memory when creating device local image because: {}", err)));
                },
            };
            std::ptr::copy_nonoverlapping(image.as_ptr(), data_ptr, image_size as usize);
            self.device.unmap_memory(staging_allocation.memory);
        };

        let mut image_allocation = self.create_image( image.dimensions().0, image.dimensions().1, mip_levels, num_samples, vk::Format::R8G8B8A8_SRGB, vk::ImageTiling::OPTIMAL, vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        match self.transition_image_layout(command_pool, graphics_queue, &image_allocation.image.unwrap(), vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL, mip_levels) {
            Ok(_) => {},
            Err(err) => {
                self.free_memory_allocation(staging_allocation)?;
                self.free_memory_allocation(image_allocation)?;
                return Err(Cow::from(format!("Failed to transition image layout when creating device local image because: {}", err)));
            },
        };
        match self.copy_buffer_to_image(&staging_allocation.buffer.unwrap(), &image_allocation.image.unwrap(), image.dimensions().0, image.dimensions().1, command_pool, graphics_queue) {
            Ok(_) => {},
            Err(err) => {
                self.free_memory_allocation(staging_allocation)?;
                self.free_memory_allocation(image_allocation)?;
                return Err(Cow::from(format!("Failed to copy buffer to image when creating device local image because: {}", err)));
            },
        };
        //Self::transition_image_layout(device, command_pool, graphics_queue, &vk_image, vk::Format::R8G8B8A8_SRGB, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, mip_levels);
        
        self.free_memory_allocation(staging_allocation)?;
        // unsafe {
        //     // device.destroy_buffer(staging_buffer, Some(&mut allocator.get_allocation_callbacks()));
        //     // device.free_memory(staging_buffer_memory, Some(&mut allocator.get_allocation_callbacks()));
        // }
        
        self.generate_mipmaps(command_pool, graphics_queue, &image_allocation.image.unwrap(), vk::Format::R8G8B8A8_SRGB, image.dimensions().0, image.dimensions().1, mip_levels)?;
        
        image_allocation.mip_levels = Some(mip_levels);

        Ok(image_allocation)
    }

    pub fn create_image_view(&mut self, allocation_info: &mut AllocationInfo, format: vk::Format, aspect_flags: vk::ImageAspectFlags, mip_levels: u32) -> Result<(), Cow<'static, str>> {
        let image = match allocation_info.image {
            Some(image) => image,
            None => return Err(Cow::from("Failed to create image view because the image was None!")),
        };
        
        let view_info = vk::ImageViewCreateInfo {
            s_type: StructureType::IMAGE_VIEW_CREATE_INFO,
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let image_view = unsafe {
            match self.device.create_image_view(&view_info, Some(&self.get_allocation_callbacks())) {
                Ok(image_view) => image_view,
                Err(err) => return Err(Cow::from(format!("Failed to create image view when creating image view because: {}", err))),
            }
        };

        allocation_info.image_view = Some(image_view);

        Ok(())
    }

    pub fn free_all_allocations(&mut self) -> Result<(), Cow<'static, str>> {
        for (_, allocations) in self.device_allocations.iter() {
            for (memory, _) in allocations.iter() {
                unsafe {
                    self.device.free_memory(*memory, Some(&self.get_allocation_callbacks()));
                }
            }
        }
        self.device_allocations.clear();
        unsafe { 
            let mut allocator = match self.host_allocator.lock() {
                Ok(allocator) => allocator,
                Err(err) => return Err(Cow::from(format!("Failed to lock host allocator when freeing all allocations because: {}", err))),
            };
            allocator.free_all_host_memory()?; 
        }
        Ok(())
    }

    fn serializable_vec_to_u8_vec<T: Serializable>(vec: &[T]) -> Vec<u8> {
        let mut u8_vec = Vec::with_capacity(vec.len() * std::mem::size_of::<T>());
        for item in vec {
            u8_vec.append(&mut item.to_u8());
        }
        u8_vec
    }

    fn generate_mipmaps(&mut self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, image: &vk::Image, image_format: vk::Format, width: u32, height: u32, mip_levels: u32) -> Result<(), Cow<'static, str>> {
        let format_properties = unsafe {
            self.instance.get_physical_device_format_properties(self.physical_device, image_format)
        };

        if !format_properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
            panic!("Texture image format does not support linear blitting!");
        }

        let command_buffer = self.begin_single_time_command(command_pool)?;
        
        let mut image_barrier = vk::ImageMemoryBarrier {
            s_type: StructureType::IMAGE_MEMORY_BARRIER,
            image: *image,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            image_barrier.subresource_range.base_mip_level = i - 1;
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                self.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[image_barrier]);
            }

            let blit = vk::ImageBlit {
                src_offsets: [
                    vk::Offset3D {
                        x: 0,
                        y: 0,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D {
                        x: 0,
                        y: 0,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            unsafe {
                self.device.cmd_blit_image(command_buffer, *image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, *image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
            }

            image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                self.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, DependencyFlags::empty(), &[], &[], &[image_barrier]);
            }

            if mip_width > 1 {
                mip_width /= 2;
            }
            if mip_height > 1 {
                mip_height /= 2;
            }
        }

        image_barrier.subresource_range.base_mip_level = mip_levels - 1;
        image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            self.device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER, DependencyFlags::empty(), &[], &[], &[image_barrier]);
        } 

        match self.end_single_time_command(command_pool, graphics_queue, command_buffer) {
            Ok(_) => {},
            Err(err) => return Err(Cow::from(format!("Failed to end single time command when generating mipmaps because: {}", err))),
        
        };
        Ok(())
    }

    fn transition_image_layout(&mut self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, image: &vk::Image, format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout, mip_levels: u32) -> Result<(), Cow<'static, str>> {
        let command_buffer = self.begin_single_time_command(command_pool)?;

        let mut barrier = vk::ImageMemoryBarrier {
            s_type: StructureType::IMAGE_MEMORY_BARRIER,
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: *image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            // src_access_mask: match old_layout { // This could cause issues, see transition barrier masks on https://vulkan-tutorial.com/Texture_mapping/Images
            //     vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
            //     vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            //     _ => panic!("Unsupported layout transition!"),
            // },
            // dst_access_mask: match new_layout {
            //     vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            //     vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
            //     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            //     _ => panic!("Unsupported layout transition!"),
            // },
            ..Default::default()
        };

        let (source_stage, destination_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => {
                barrier.src_access_mask = vk::AccessFlags::empty();
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER)
            },
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => {
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
                (vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER)
            },
            //(vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS),
            _ => panic!("Unsupported layout transition! {} {}", old_layout.as_raw(), new_layout.as_raw()),
        };

        unsafe {
            self.device.cmd_pipeline_barrier(command_buffer, source_stage, destination_stage, vk::DependencyFlags::empty(), &[], &[], &[barrier]);
        }

        match self.end_single_time_command(command_pool, graphics_queue, command_buffer) {
            Ok(_) => {},
            Err(err) => return Err(Cow::from(format!("Failed to end single time command when transitioning image layout because: {}", err))),
        
        };
        Ok(())
    }

    pub fn free_memory_allocation(&mut self, allocation_info: AllocationInfo) -> Result<(), Cow<'static, str>> {
        if let Some(memories) = self.device_allocations.get_mut(&allocation_info.memory_index) {
            for (memory, free_ranges) in memories.iter_mut() {
                if *memory != allocation_info.memory {
                    continue;
                }
                free_ranges.push((allocation_info.memory_start, allocation_info.memory_end));
                
                free_ranges.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                let mut i = 0;
                while i < free_ranges.len() - 1 {
                    if free_ranges[i].1 == free_ranges[i + 1].0 - 1 {
                        free_ranges[i].1 = free_ranges[i + 1].1;
                        free_ranges.remove(i + 1);
                    }
                    i += 1;
                }
            }


            if let Some(buffer) = allocation_info.buffer {
                unsafe {
                    self.device.destroy_buffer(buffer, Some(&self.get_allocation_callbacks()));
                }
            }
            if let Some(image_view) = allocation_info.image_view {
                unsafe {
                    self.device.destroy_image_view(image_view, Some(&self.get_allocation_callbacks()));
                }
            }
            if let Some(image) = allocation_info.image {
                unsafe {
                    self.device.destroy_image(image, Some(&self.get_allocation_callbacks()));
                }
            }
        } else {
            return Err(Cow::from("Failed to free memory!"));
        }
        Ok(())
    }

    fn copy_buffer_to_image(&self, src_buffer: &vk::Buffer, dst_image: &vk::Image, width: u32, height: u32, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue) -> Result<(), Cow<'static, str>> {
        let command_buffer = self.begin_single_time_command(command_pool)?;

        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: 0,
                y: 0,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };

        unsafe {
            self.device.cmd_copy_buffer_to_image(command_buffer, *src_buffer, *dst_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        }

        self.end_single_time_command(command_pool, graphics_queue, command_buffer)?;
        Ok(())
    }

    fn copy_buffer(&self, src_allocation: &AllocationInfo, dst_allocation: &AllocationInfo, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue) -> Result<(), Cow<'static, str>> {
        let command_buffer = self.begin_single_time_command(command_pool)?;

        let size = src_allocation.memory_end - src_allocation.memory_start;

        let copy_region = vk::BufferCopy {
            size,
            ..Default::default()
        };

        let Some(src_buffer) = src_allocation.buffer else {
            return Err(Cow::from("Failed to copy buffer because the src buffer was None!"));
        };

        let Some(dst_buffer) = dst_allocation.buffer else {
            return Err(Cow::from("Failed to copy buffer because the dst buffer was None!"));
        };

        unsafe {
            self.device.cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, &[copy_region]);
        }

        self.end_single_time_command(command_pool, graphics_queue, command_buffer)?;

        Ok(())
    }

    fn begin_single_time_command(&self, command_pool: &vk::CommandPool) -> Result<vk::CommandBuffer, Cow<'static, str>> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: *command_pool,
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffer = unsafe {
            let allocated_command_buffers = match self.device.allocate_command_buffers(&alloc_info) {
                Ok(command_buffers) => command_buffers,
                Err(err) => return Err(Cow::from(format!("Failed to allocate command buffer when beginning single time command because: {}", err))),
            };
            allocated_command_buffers[0]
        };

        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            match self.device.begin_command_buffer(command_buffer, &begin_info) {
                Ok(_) => {},
                Err(err) => {
                    self.device.free_command_buffers(*command_pool, &[command_buffer]);
                    return Err(Cow::from(format!("Failed to begin command buffer when beginning single time command because: {}", err)))
                },
            };
        }

        Ok(command_buffer)
    }

    fn end_single_time_command(&self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, command_buffer: vk::CommandBuffer) -> Result<(), Cow<'static, str>> {
        unsafe {
            match self.device.end_command_buffer(command_buffer) {
                Ok(_) => {},
                Err(err) => {
                    self.device.free_command_buffers(*command_pool, &[command_buffer]);
                    return Err(Cow::from(format!("Failed to end command buffer when ending single time command because: {}", err)))
                },
            }
        }

        let submit_info = vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };

        unsafe {
            match self.device.queue_submit(*graphics_queue, &[submit_info], vk::Fence::null()) {
                Ok(_) => {},
                Err(err) => {
                    self.device.free_command_buffers(*command_pool, &[command_buffer]);
                    return Err(Cow::from(format!("Failed to submit queue when ending single time command because: {}", err)))
                },
            };
            match self.device.queue_wait_idle(*graphics_queue) {
                Ok(_) => {},
                Err(err) => {
                    self.device.free_command_buffers(*command_pool, &[command_buffer]);
                    return Err(Cow::from(format!("Failed to wait for queue to be idle when ending single time command because: {}", err)))
                },
            };
            self.device.free_command_buffers(*command_pool, &[command_buffer]);
        }

        Ok(())
    }

    fn allocate_new_device_memory(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize, force_own_memory_block: bool) -> Result<(), Cow<'static, str>> {
        let allocated_size = size.max(Self::DEFAULT_DEVICE_MEMORY_ALLOCATION_BYTE_SIZE) * !force_own_memory_block as vk::DeviceSize + force_own_memory_block as vk::DeviceSize * size;
        
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: allocated_size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe {
            match self.device.allocate_memory(&alloc_info, Some(&self.get_allocation_callbacks())) {
                Ok(memory) => memory,
                Err(err) => return Err(Cow::from(format!("Failed to allocate memory when allocating new device memory because: {}", err))),
            }
        };

        self.device_allocations.entry(memory_type_index).or_default().push((memory, vec![(0, allocated_size)]));
        Ok(())
    }

    fn get_allocation(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize, alignment: vk::DeviceSize, force_own_memory_block: bool) -> Result<AllocationInfo, Cow<'static, str>> {
        let mut allocation = self.find_allocation(memory_type_index, size, alignment);

        if allocation.is_err() {
            self.allocate_new_device_memory(memory_type_index, size, force_own_memory_block)?;
            allocation = self.find_allocation(memory_type_index, size, alignment);
        }

        allocation
    }

    fn find_allocation(&mut self, memory_type_index: u32, size: u64, alignment: vk::DeviceSize) -> Result<AllocationInfo, Cow<'static, str>> {
        if let Some(memories) = self.device_allocations.get_mut(&memory_type_index) {
            for (memory, free_ranges) in memories.iter_mut() {
                for (start, end) in free_ranges.iter_mut() {
                    let alignment_offset = if *start % alignment == 0 { 0 } else { alignment - (*start % alignment) };
                    let aligned_start = *start + alignment_offset;
                    if *end - aligned_start >= size {
                        let allocation = Ok(AllocationInfo {
                            memory_index: memory_type_index,
                            memory_start: aligned_start,
                            memory_end: aligned_start + size,
                            buffer: None,
                            image: None,
                            memory: *memory,
                            image_view: None,
                            uniform_pointers: Vec::new(),
                            mip_levels: None,
                        });
                        *start += size + alignment_offset;
                        return allocation;
                    }
                }
            }
        }
        Err(Cow::from("Failed to find allocation!"))
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, Cow<'static, str>> {
        let mem_properties = unsafe {
            self.instance.get_physical_device_memory_properties(self.physical_device)
        };

        for (i, mem_type) in mem_properties.memory_types.iter().enumerate() {
            if type_filter & (1 << i) != 0 && mem_type.property_flags.contains(properties) {
                return Ok(i as u32);
            }
        }
        Err(Cow::from("Failed to find suitable memory type!"))
    }

    pub unsafe fn get_allocation_callbacks(&self) -> vk::AllocationCallbacks {
        let callbacks = vk::AllocationCallbacks {
            p_user_data: Arc::into_raw(self.host_allocator.clone()) as *mut c_void,
            pfn_allocation: Some(pfn_allocation),
            pfn_reallocation: Some(pfn_reallocation),
            pfn_free: Some(pfn_free),
            pfn_internal_allocation: None,
            pfn_internal_free: None,
        };
        callbacks
    }
}

impl AllocationInfo {
    pub fn get_memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    pub fn get_buffer(&self) -> Option<vk::Buffer> {
        self.buffer
    }

    pub fn get_image(&self) -> Option<vk::Image> {
        self.image
    }

    pub fn get_image_view(&self) -> Option<vk::ImageView> {
        self.image_view
    }

    pub fn get_memory_start(&self) -> vk::DeviceSize {
        self.memory_start
    }

    pub fn get_uniform_pointers(&self) -> &[*mut c_void] {
        &self.uniform_pointers
    }

    pub fn get_mip_levels(&self) -> Option<u32> {
        self.mip_levels
    }
}

// Host memory allocation
impl VkHostAllocator {
    const DEFAULT_HOST_MEMORY_ALLOCATION_BYTE_SIZE: usize = 512_000; // 512 KB

    pub fn allocate_host_memory(&mut self, size: usize, alignment: usize) -> Result<*mut c_void, Cow<'static, str>> {
        let mut allocation = self.find_host_allocation(size, alignment);

        if allocation.is_err() {
            unsafe {
                self.allocate_new_host_memory(size, alignment)?;
            }
            allocation = self.find_host_allocation(size, alignment);
        }

        allocation
    }

    fn find_host_allocation(&mut self, size: usize, alignment: usize) -> Result<*mut c_void, Cow<'static, str>> {
        if let Some(allocations) = self.host_allocations.get_mut(&alignment) {
            for allocation in allocations.iter_mut() {
                for free_range in allocation.free_allocations.iter_mut() {
                    if (free_range.1 + 1) - free_range.0 >= size {
                        let allocation_ptr = unsafe { allocation.start_ptr.add(free_range.0) as *mut c_void};
                        let previous = self.allocated_host_pointers.get(&allocation_ptr);
                        if previous.is_some() {
                            return Err(Cow::from("Failed to find host allocation! Because the allocation was already allocated!"));
                        }
                        self.allocated_host_pointers.insert(allocation_ptr, (alignment, size));
                        // Add size and padding to allocation, so that the alignment is correct for the next allocation as well
                        free_range.0 += size + ((alignment - (size % alignment)) % alignment);
                        return Ok(allocation_ptr);
                    }
                }
            }
        }
        Err(Cow::from("Failed to find host allocation!"))
    }

    unsafe fn allocate_new_host_memory(&mut self, size: usize, alignment: usize) -> Result<(), Cow<'static, str>> {
        let allocated_size = size.max(Self::DEFAULT_HOST_MEMORY_ALLOCATION_BYTE_SIZE).div_ceil(alignment) * alignment;

        let layout = match std::alloc::Layout::from_size_align(allocated_size, alignment) {
            Ok(layout) => layout,
            Err(err) => return Err(Cow::from(format!("Failed to create layout when allocating new host memory because: {}", err))),
        
        };

        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            return Err(Cow::from("Failed to allocate new host memory!"));
        }

        let allocation = HostAllocationPool {
            start_ptr: ptr,
            size: allocated_size,
            alignment,
            free_allocations: vec![(0, allocated_size - 1)],
        };
        self.host_allocations.entry(alignment).or_default().push(allocation);
        Ok(())
    }

    pub unsafe fn free_host_memory(&mut self, ptr: *mut c_void) -> Result<(), Cow<'static, str>> {
        if let Some((alignment, size)) = self.allocated_host_pointers.remove(&ptr) {
            if let Some(allocations) = self.host_allocations.get_mut(&alignment) {
                for allocation in allocations.iter_mut() {
                    if allocation.start_ptr <= ptr as *mut u8 && allocation.start_ptr.add(allocation.size) > ptr as *mut u8 {
                        let pointer_offset: usize = match ptr.offset_from(allocation.start_ptr as *mut c_void).try_into() {
                            Ok(offset) => offset,
                            Err(err) => return Err(Cow::from(format!("Failed to free host memory because: {}", err))),
                        };
                        allocation.free_allocations.push((pointer_offset, pointer_offset + size + ((alignment - (size % alignment)) % alignment) - 1));
                        allocation.free_allocations.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                        let mut i = 0;
                        while i < allocation.free_allocations.len() - 1 {
                            if allocation.free_allocations[i].1 == allocation.free_allocations[i + 1].0 - 1 {
                                allocation.free_allocations[i].1 = allocation.free_allocations[i + 1].1;
                                allocation.free_allocations.remove(i + 1);
                            }
                            i += 1;
                        }
                        return Ok(());
                    }
                }
            }
        }

        Err(Cow::from("Failed to free host memory!"))
    }

    pub unsafe fn free_all_host_memory(&mut self) -> Result<(), Cow<'static, str>> {
        for (_, allocations) in self.host_allocations.iter_mut() {
            for allocation in allocations.iter_mut() {
                let layout = match std::alloc::Layout::from_size_align(allocation.size, allocation.alignment) {
                    Ok(layout) => layout,
                    Err(err) => return Err(Cow::from(format!("Failed to create layout when freeing all host memory because: {}", err))),
                };
                std::alloc::dealloc(allocation.start_ptr, layout);
            }
        }
        self.host_allocations.clear();
        self.allocated_host_pointers.clear();
        Ok(())
    }

    pub unsafe fn reallocate(&mut self, ptr: *mut c_void, new_size: usize) -> Result<*mut c_void, Cow<'static, str>> {
        if let Some((alignment, size)) = self.allocated_host_pointers.get(&ptr) {
            let new_ptr = self.allocate_host_memory(new_size, *alignment)?;
            std::ptr::copy_nonoverlapping(ptr, new_ptr, new_size);
            self.free_host_memory(ptr)?;
            return Ok(new_ptr);
        }
        Err(Cow::from("Failed to reallocate host memory!"))
    }
}

unsafe extern "system" fn pfn_allocation(p_user_data: *mut c_void, size: usize, alignment: usize, allocation_scope: SystemAllocationScope) -> *mut c_void {
    let allocator_arc = Arc::from_raw(p_user_data as *mut Mutex<VkHostAllocator>);
    
    let alloced_ptr = {
        let allocator = &mut allocator_arc.lock().unwrap();

        match allocation_scope {
            SystemAllocationScope::COMMAND => {
                match allocator.allocate_host_memory(size, alignment) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to allocate host memory when allocating command because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::OBJECT => {
                match allocator.allocate_host_memory(size, alignment) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to allocate host memory when allocating object because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::CACHE => {
                match allocator.allocate_host_memory(size, alignment) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to allocate host memory when allocating cache because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::DEVICE => {
                match allocator.allocate_host_memory(size, alignment) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to allocate host memory when allocating device because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::INSTANCE => {
                match allocator.allocate_host_memory(size, alignment) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to allocate host memory when allocating instance because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            _ => {
                eprintln!("Failed to allocate host memory because the allocation scope was not supported!");
                std::ptr::null_mut()
            },
        }
    };

    std::mem::forget(allocator_arc);
    alloced_ptr
}

unsafe extern "system" fn pfn_reallocation(p_user_data: *mut c_void, original: *mut c_void, size: usize, alignment: usize, allocation_scope: SystemAllocationScope) -> *mut c_void {
    let allocator_arc = Arc::from_raw(p_user_data as *mut Mutex<VkHostAllocator>);
    
    let realloc_ptr = {
        let allocator = &mut allocator_arc.lock().unwrap();
        match allocation_scope {
            SystemAllocationScope::COMMAND => {
                match allocator.reallocate(original, size) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to reallocate host memory when allocating command because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::OBJECT => {
                match allocator.reallocate(original, size) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to reallocate host memory when allocating object because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::CACHE => {
                match allocator.reallocate(original, size) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to reallocate host memory when allocating cache because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::DEVICE => {
                match allocator.reallocate(original, size) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to reallocate host memory when allocating device because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            SystemAllocationScope::INSTANCE => {
                match allocator.reallocate(original, size) {
                    Ok(ptr) => ptr,
                    Err(err) => {
                        eprintln!("Failed to reallocate host memory when allocating instance because: {}", err);
                        std::ptr::null_mut()
                    },
                }
            },
            _ => {
                eprintln!("Failed to reallocate host memory because the allocation scope was not supported!");
                std::ptr::null_mut()
            },
        }
    };
    
    std::mem::forget(allocator_arc);
    realloc_ptr
}

unsafe extern "system" fn pfn_free(p_user_data: *mut c_void, ptr: *mut c_void) {
    if ptr.is_null() {
        return;
    }

    let allocator_arc = Arc::from_raw(p_user_data as *mut Mutex<VkHostAllocator>);
    {
        let allocator = &mut allocator_arc.lock().unwrap();
        match allocator.free_host_memory(ptr) {
            Ok(_) => {},
            Err(err) => {
                eprintln!("Failed to free host memory when freeing because: {}", err);
            },
        };
    }

    std::mem::forget(allocator_arc);
}
