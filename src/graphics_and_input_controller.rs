use winit::{event_loop::EventLoop, window::WindowBuilder};

use crate::vk_controller::VkController;



pub struct GraphicsAndInputController {
    vk_controller: VkController,
    event_loop: EventLoop<()>,
}

impl GraphicsAndInputController {
    pub fn new(window_title: &str, application_name: &str) -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().with_title(window_title).build(&event_loop).unwrap();
        let vk_controller = VkController::new(window, application_name);

        Self {
            vk_controller,
            event_loop,
        }
    }


    pub fn run(&mut self, ) {
        
    }
}