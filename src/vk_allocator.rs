use ash::{vk, Instance, Device};


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

    
}