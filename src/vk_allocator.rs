use std::{borrow::Cow, collections::HashMap};

use ash::{vk::{self, StructureType}, Instance, Device};

type MemoryTypeIndex = u32;
type MemoryOffset = vk::DeviceSize;

pub struct AllocationInfo {
    buffer: Option<vk::Buffer>,
    image: Option<vk::Image>,
    memory_index: MemoryTypeIndex,
    memory_start: MemoryOffset,
    memory_end: vk::DeviceSize,
}

pub struct VkAllocator {
    device: Device,
    physical_device: vk::PhysicalDevice,
    instance: Instance,
    allocations: HashMap<MemoryTypeIndex, Vec<(vk::DeviceMemory, Vec<(vk::DeviceSize, vk::DeviceSize)>)>>,
}

impl VkAllocator {
    const DEFAULT_MEMORY_BYTE_SIZE: vk::DeviceSize = 256_000_000; // 256 MB 

    pub fn new(device: Device, physical_device: vk::PhysicalDevice, instance: Instance) -> Self {
        Self {
            device,
            physical_device,
            instance,
            allocations: HashMap::new(),
        }
    }

    pub fn create_buffer(&mut self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> AllocationInfo {
        let buffer_info = vk::BufferCreateInfo {
            s_type: StructureType::BUFFER_CREATE_INFO,
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None)
        }.unwrap();
        
        let memory_requirements = unsafe {
            self.device.get_buffer_memory_requirements(buffer)
        };

        let alloc_info = vk::MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: memory_requirements.size,
            memory_type_index: self.find_memory_type( memory_requirements.memory_type_bits, properties).unwrap(),
            ..Default::default()
        };

        let (buffer_memory, mut allocation_info) = self.get_allocation(alloc_info.memory_type_index, alloc_info.allocation_size).unwrap();

        unsafe {
            self.device.bind_buffer_memory(buffer, buffer_memory, allocation_info.memory_start).unwrap();
        }

        allocation_info.buffer = Some(buffer);

        allocation_info
    }

    fn allocate_new_device_memory(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize) {
        let allocated_size = size.max(Self::DEFAULT_MEMORY_BYTE_SIZE);
        
        let alloc_info = vk::MemoryAllocateInfo {
            s_type: StructureType::MEMORY_ALLOCATE_INFO,
            allocation_size: allocated_size,
            memory_type_index,
            ..Default::default()
        };

        let memory = unsafe {
            self.device.allocate_memory(&alloc_info, None)
        }.unwrap();

        self.allocations.entry(memory_type_index).or_default().push((memory, vec![(0, allocated_size)]));
    }

    fn get_allocation(&mut self, memory_type_index: MemoryTypeIndex, size: vk::DeviceSize) -> Option<(vk::DeviceMemory, AllocationInfo)> {
        let mut allocation = self.find_allocation(memory_type_index, size);

        if allocation.is_none() {
            self.allocate_new_device_memory(memory_type_index, size);
            allocation = self.find_allocation(memory_type_index, size);
        }

        allocation
    }

    fn find_allocation(&mut self, memory_type_index: u32, size: u64) -> Option<(vk::DeviceMemory, AllocationInfo)> {
        if let Some(memories) = self.allocations.get_mut(&memory_type_index) {
            for (memory, free_ranges) in memories.iter_mut() {
                for (start, end) in free_ranges.iter_mut() {
                    if *end - *start >= size {
                        let allocation = Some((*memory, AllocationInfo { // TODO handle adding buffer and image to allocation info when actually making them
                            memory_index: memory_type_index,
                            memory_start: *start,
                            memory_end: *start + size - 1,
                            buffer: None,
                            image: None, 
                        }));
                        *start += size;
                        return allocation;
                    }
                }
            }
        }
        None
    }

    pub fn free_memory_allocation(&mut self, allocation_info: AllocationInfo) -> Result<(), Cow<'static, str>> {
        if let Some(memories) = self.allocations.get_mut(&allocation_info.memory_index) {
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
        } else {
            return Err(Cow::from("Failed to free memory!"));
        }
        Ok(())
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
