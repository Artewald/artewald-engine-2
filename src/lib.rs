use winit::{event_loop::EventLoop, window::WindowBuilder};

pub mod graphics_objects;
mod object_manager;
pub mod pipeline_manager;
mod sampler_manager;
mod vertex;
mod vk_allocator;
mod vk_controller;
pub mod artewald_engine;
pub mod inputs;


pub fn create_new_renderer(window_title: &str, application_name: &str) -> vk_controller::VkController {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title(window_title).build(&event_loop).unwrap();

    vk_controller::VkController::new(window, application_name)
}
