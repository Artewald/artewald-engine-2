use std::{sync::{Arc, RwLock}, time::Instant};

use winit::{event::{ElementState, Event, KeyboardInput, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::WindowBuilder};

use crate::vk_controller::VkController;



pub struct GraphicsAndInputController {
    vk_controller: VkController,
    event_loop: Option<EventLoop<()>>,
    currently_pressed_keys: Vec<winit::event::VirtualKeyCode>,
}

impl GraphicsAndInputController {
    pub fn new(window_title: &str, application_name: &str) -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().with_title(window_title).build(&event_loop).unwrap();
        let vk_controller = VkController::new(window, application_name);

        Self {
            vk_controller,
            event_loop: Some(event_loop),
            currently_pressed_keys: Vec::new(),
        }
    }

    // Add on key pressed callback
    

    // Add on key released callback


    // Add on key held callback


    // Add on mouse button pressed callback


    // Add on mouse button released callback


    // Add on mouse button held callback


    // Add on mouse moved callback (here we should not use winit's event loop, but instead use some other crate, due to the docs telling that we shoud not use it for 3d stuff)


    // Add on mouse scrolled callback

    pub fn run(controller: Arc<RwLock<Self>>, new_main_function: fn(Arc<RwLock<Self>>)) {
        let event_loop = controller.write().unwrap().event_loop.take().unwrap();
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
                        {
                            let mut controller_lock = controller.write().unwrap();
                            controller_lock.vk_controller.frame_buffer_resized = true;
                        }
                    },
                    WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            // state: ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        },
                        ..
                    } => {
                        match keycode {
                            winit::event::VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            },
                            // winit::event::VirtualKeyCode::Key1 => {
                            //     vk_controller.remove_object_to_render(current_object_id);
                            //     current_object_id = vk_controller.add_object_to_render(obj_one.clone()).unwrap();
                            // },
                            // winit::event::VirtualKeyCode::Key2 => {
                            //     vk_controller.remove_object_to_render(current_object_id);
                            //     current_object_id = vk_controller.add_object_to_render(obj_two.clone()).unwrap();
                            // },
                            // winit::event::VirtualKeyCode::Key3 => {
                            //     vk_controller.remove_object_to_render(current_object_id);
                            //     current_object_id = vk_controller.add_object_to_render(obj_three.clone()).unwrap();
                            // }
                            _ => {}
                        }
                    },
                    _ => {}
                },
                Event::LoopDestroyed => {
                    {
                        let mut controller_lock = controller.write().unwrap();
                        controller_lock.vk_controller.cleanup();
                    }
                    close = true;
                }
                _ => {}
            }

            if close {
                return;
            }
            
            {
                let mut controller_lock = controller.write().unwrap();
                if controller_lock.vk_controller.try_to_draw_frame() {
                    frame_count += 1;
                    if last_fps_print.elapsed().as_secs_f32() > 1.0 {
                        println!("FPS: {}", frame_count as f32 / last_fps_print.elapsed().as_secs_f32());
                        frame_count = 0;
                        last_fps_print = std::time::Instant::now();
                    }
                }
            }
    
        });
    }
}