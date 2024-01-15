use std::borrow::Cow;

use ash::{vk::{self, StructureType}, Instance, Device};


pub struct VkAllocator {
    device: Device,
    physical_device: vk::PhysicalDevice,
    instance: Instance,
}

impl VkAllocator {
    pub fn new(device: Device, physical_device: vk::PhysicalDevice, instance: Instance) -> Self {
        Self {
            device,
            physical_device,
            instance,
        }
    }

    pub fn create_buffer(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> (vk::Buffer, vk::DeviceMemory) {
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

        println!("Should not allocate memory for many different buffers, but rather have one big buffer and an allocator that uses offsets when binding buffer to memory!");
        let buffer_memory = unsafe {
            self.device.allocate_memory(&alloc_info, None)
        }.unwrap();

        unsafe {
            self.device.bind_buffer_memory(buffer, buffer_memory, 0).unwrap();
        }

        (buffer, buffer_memory)
    }

    pub fn create_device_buffer

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