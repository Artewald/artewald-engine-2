use std::borrow::Cow;

use ash::{vk::{self, Sampler}, Device, Instance};

use crate::vk_allocator::VkAllocator;

pub struct SamplerConfig {
    pub s_type: vk::StructureType,
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub anisotropy_enable: vk::Bool32,
    // pub max_anisotropy: f32,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: vk::Bool32,
    pub compare_enable: vk::Bool32,
    pub compare_op: vk::CompareOp,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub mip_lod_bias: f32,
    pub min_lod: f32,
    pub max_lod: f32,
}

pub struct SamplerManager {
    samplers: Vec<(SamplerConfig, Sampler)>
}

impl SamplerManager {
    pub fn new() -> Self {
        Self {
            samplers: Vec::new()
        }
    }

    pub fn get_or_create_sampler(&mut self, device: &Device, instance: &Instance, physical_device: &vk::PhysicalDevice, sampler_config: SamplerConfig, allocator: &mut VkAllocator) -> Result<Sampler, Cow<'static, str>> {
        for (config, sampler) in &self.samplers {
            if config == &sampler_config {
                return Ok(*sampler);
            }
        }

        let max_anisotropy = unsafe {
            instance.get_physical_device_properties(*physical_device).limits.max_sampler_anisotropy
        };

        let sampler_create_info = vk::SamplerCreateInfo {
            s_type: sampler_config.s_type,
            mag_filter: sampler_config.mag_filter,
            min_filter: sampler_config.min_filter,
            address_mode_u: sampler_config.address_mode_u,
            address_mode_v: sampler_config.address_mode_v,
            address_mode_w: sampler_config.address_mode_w,
            anisotropy_enable: sampler_config.anisotropy_enable,
            max_anisotropy,
            border_color: sampler_config.border_color,
            unnormalized_coordinates: sampler_config.unnormalized_coordinates,
            compare_enable: sampler_config.compare_enable,
            compare_op: sampler_config.compare_op,
            mipmap_mode: sampler_config.mipmap_mode,
            mip_lod_bias: sampler_config.mip_lod_bias,
            min_lod: sampler_config.min_lod,
            max_lod: sampler_config.max_lod,
            ..Default::default()
        };

        let sampler = unsafe {
            device.create_sampler(&sampler_create_info, Some(&allocator.get_allocation_callbacks()))
        }.map_err(|err| Cow::Owned(format!("Failed to create sampler: {}", err)))?;

        self.samplers.push((sampler_config, sampler));
        return Ok(sampler);
    }

    pub fn destroy_samplers(&mut self, device: &Device, allocator: &mut VkAllocator) {
        for (_, sampler) in self.samplers.drain(..) {
            unsafe {
                device.destroy_sampler(sampler, Some(&allocator.get_allocation_callbacks()));
            }
        }
    }
} 

impl Eq for SamplerConfig { }

impl PartialEq for SamplerConfig {
    fn eq(&self, other: &Self) -> bool {
        self.s_type == other.s_type &&
        self.mag_filter == other.mag_filter &&
        self.min_filter == other.min_filter &&
        self.address_mode_u == other.address_mode_u &&
        self.address_mode_v == other.address_mode_v &&
        self.address_mode_w == other.address_mode_w &&
        self.anisotropy_enable == other.anisotropy_enable &&
        // self.max_anisotropy == other.max_anisotropy &&
        self.border_color == other.border_color &&
        self.unnormalized_coordinates == other.unnormalized_coordinates &&
        self.compare_enable == other.compare_enable &&
        self.compare_op == other.compare_op &&
        self.mipmap_mode == other.mipmap_mode &&
        self.mip_lod_bias == other.mip_lod_bias &&
        self.min_lod == other.min_lod &&
        self.max_lod == other.max_lod
    }
}