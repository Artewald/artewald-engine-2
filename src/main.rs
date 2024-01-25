use vertex::Vertex;
use vk_controller::VkController;
use winit::{event_loop::{EventLoop, ControlFlow}, window::WindowBuilder, event::{Event, WindowEvent, ElementState, KeyboardInput}};
use nalgebra_glm as glm;

mod vk_controller;
mod vertex;
mod graphics_objects;
mod vk_allocator;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Artewald Engine 2").build(&event_loop).unwrap();

    let mut vk_controller = VkController::new(window, "Artewald Engine 2");

    let mut frame_count = 0;
    let mut last_fps_print = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let mut close = false;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                },
                WindowEvent::Resized(_) => {
                    vk_controller.frame_buffer_resized = true;
                },
                WindowEvent::KeyboardInput {
                    input: KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                    ..
                } => {
                    match keycode {
                        winit::event::VirtualKeyCode::Escape => {
                            *control_flow = ControlFlow::Exit;
                        },
                        _ => {}
                    }
                },
                _ => {}
            },
            Event::LoopDestroyed => {
                vk_controller.cleanup();
                close = true;
            }
            _ => {}
        }

        if close {
            return;
        }

        vk_controller.draw_frame();
        frame_count += 1;
        if last_fps_print.elapsed().as_secs_f32() > 1.0 {
            println!("FPS: {}", frame_count as f32 / last_fps_print.elapsed().as_secs_f32());
            frame_count = 0;
            last_fps_print = std::time::Instant::now();
        }
    });
}
