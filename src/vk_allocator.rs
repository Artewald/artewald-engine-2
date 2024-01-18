use std::{borrow::Cow, collections::HashMap, rc::Rc, ffi::c_void};

use ash::{vk::{self, StructureType, SystemAllocationScope}, Instance, Device};

type MemoryTypeIndex = u32;
type MemoryOffset = vk::DeviceSize;
type MemorySizeRange = (vk::DeviceSize, vk::DeviceSize);
type Alignment = usize;

pub struct AllocationInfo {
    buffer: Option<vk::Buffer>,
    image: Option<vk::Image>,
    memory_index: MemoryTypeIndex,
    memory_start: MemoryOffset,
    memory_end: vk::DeviceSize,
    memory: vk::DeviceMemory,
}

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
            host_allocations: HashMap::new(),
            allocated_host_pointers: HashMap::new(),
        }
    }

    pub fn create_buffer(&mut self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<AllocationInfo, Cow<'static, str>> {
        let buffer_info = vk::BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            match self.device.create_buffer(&buffer_info, None) {
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

        let (buffer_memory, mut allocation_info) = self.get_allocation(alloc_info.memory_type_index, alloc_info.allocation_size)?;

        unsafe {
            match self.device.bind_buffer_memory(buffer, buffer_memory, allocation_info.memory_start) {
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

    pub fn create_device_local_buffer(&mut self, command_pool: &vk::CommandPool, graphics_queue: &vk::Queue, data: &[u8], buffer_usage: vk::BufferUsageFlags) -> Result<AllocationInfo, Cow<'static, str>> {
        let size = std::mem::size_of_val(data);
        
        let staging_allocation = self.create_buffer(size as u64, vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)?;
        
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
        
        let device_local_allocation = self.create_buffer(size as u64, buffer_usage | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

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
            match self.device.create_image(&image_info, None) {
                Ok(image) => image,
                Err(err) => return Err(Cow::from(format!("Failed to create image when creating image because: {}", err))),
            }
        };

        let mem_requirements = unsafe {
            self.device.get_image_memory_requirements(image)
        };

        let (image_memory, mut image_allocation) = self.get_allocation(self.find_memory_type(mem_requirements.memory_type_bits, properties)?, mem_requirements.size)?;

        image_allocation.image = Some(image);

        unsafe {
            self.device.bind_image_memory(image, image_memory, image_allocation.memory_start).unwrap();
        }

        Ok(image_allocation)
    }    

    pub fn free_memory_allocation(&mut self, allocation_info: AllocationInfo) -> Result<(), Cow<'static, str>> {
        if let Some(memories) = self.device_allocations.get_mut(&allocation_info.memory_index) {
            for (_, free_ranges) in memories.iter_mut() {
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
                    self.device.destroy_buffer(buffer, None);
                }
            }
            else if let Some(image) = allocation_info.image {
                unsafe {
                    self.device.destroy_image(image, None);
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

    fn allocate_new_device_memory(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize) -> Result<(), Cow<'static, str>> {
        let allocated_size = size.max(Self::DEFAULT_DEVICE_MEMORY_ALLOCATION_BYTE_SIZE);
        
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: allocated_size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe {
            match self.device.allocate_memory(&alloc_info, None) {
                Ok(memory) => memory,
                Err(err) => return Err(Cow::from(format!("Failed to allocate memory when allocating new device memory because: {}", err))),
            }
        };

        self.device_allocations.entry(memory_type_index).or_default().push((memory, vec![(0, allocated_size)]));
        Ok(())
    }

    fn get_allocation(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize) -> Result<(vk::DeviceMemory, AllocationInfo), Cow<'static, str>> {
        let mut allocation = self.find_allocation(memory_type_index, size);

        if allocation.is_err() {
            self.allocate_new_device_memory(memory_type_index, size)?;
            allocation = self.find_allocation(memory_type_index, size);
        }

        allocation
    }

    fn find_allocation(&mut self, memory_type_index: u32, size: u64) -> Result<(vk::DeviceMemory, AllocationInfo), Cow<'static, str>> {
        if let Some(memories) = self.device_allocations.get_mut(&memory_type_index) {
            for (memory, free_ranges) in memories.iter_mut() {
                for (start, end) in free_ranges.iter_mut() {
                    if *end - *start >= size {
                        let allocation = Ok((*memory, AllocationInfo { // TODO handle adding image to allocation info when actually making them
                            memory_index: memory_type_index,
                            memory_start: *start,
                            memory_end: *start + size - 1,
                            buffer: None,
                            image: None,
                            memory: *memory, 
                        }));
                        *start += size;
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
}

// Host memory allocation
impl VkAllocator {
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
                    if free_range.1 - free_range.0 >= size {
                        let allocation_ptr = unsafe { allocation.start_ptr.offset(free_range.0 as isize) as *mut c_void};
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
        let allocated_size = size.max(Self::DEFAULT_HOST_MEMORY_ALLOCATION_BYTE_SIZE);

        let layout = std::alloc::Layout::from_size_align(allocated_size, alignment).unwrap();

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
                    if allocation.start_ptr == ptr as *mut u8 {
                        allocation.free_allocations.push((0, size - 1));
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
                std::alloc::dealloc(allocation.start_ptr, std::alloc::Layout::from_size_align(allocation.size, allocation.alignment).unwrap());
            }
        }
        self.host_allocations.clear();
        self.allocated_host_pointers.clear();
        Ok(())
    }

    pub unsafe fn get_allocation_callbacks(&mut self) -> vk::AllocationCallbacks {
        let callbacks = vk::AllocationCallbacks {
            p_user_data: self as *mut VkAllocator as *mut c_void,
            pfn_allocation: Some(pfn_allocation),
            pfn_reallocation: None,
            pfn_free: None,
            pfn_internal_allocation: None,
            pfn_internal_free: None,
        };
        callbacks
    }
}

unsafe extern "system" fn pfn_allocation(p_user_data: *mut c_void, size: usize, alignment: usize, allocation_scope: SystemAllocationScope) -> *mut c_void {
    let allocator = &mut *(p_user_data as *mut VkAllocator);
    
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
}