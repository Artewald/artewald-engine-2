use std::sync::{Arc, RwLock, mpsc};

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::WindowBuilder};

use crate::{inputs::{KeyType, MouseScrollDelta, WindowPixelPosition}, vk_controller::VkController};

pub type TerminatorSignalSender = mpsc::Sender<()>;
pub type TerminatorSignalReceiver = mpsc::Receiver<()>;

pub struct ArtewaldEngine<F, G, H> where F: Fn(KeyType) + 'static, G: Fn(WindowPixelPosition) + 'static, H: Fn(MouseScrollDelta) + 'static {
    vk_controller: VkController,
    event_loop: Option<EventLoop<()>>,
    currently_pressed_keys: Vec<KeyType>,
    on_button_pressed_callback: Option<F>,
    on_button_released_callback: Option<F>,
    on_button_held_callback: Option<F>,
    on_mouse_moved_callback: Option<G>,
    on_mouse_scrolled_callback: Option<H>,
    is_cursor_locked: bool,
}

impl<F: Fn(KeyType)+'static, G: Fn(WindowPixelPosition) + 'static, H: Fn(MouseScrollDelta) + 'static> ArtewaldEngine<F, G, H> {
    pub fn new(window_title: &str, application_name: &str) -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new().with_title(window_title).build(&event_loop).unwrap();
        let vk_controller = VkController::new(window, application_name);

        Self {
            vk_controller,
            event_loop: Some(event_loop),
            currently_pressed_keys: Vec::new(),
            on_button_pressed_callback: None,
            on_button_released_callback: None,
            on_button_held_callback: None,
            on_mouse_moved_callback: None,
            on_mouse_scrolled_callback: None,
            is_cursor_locked: false,
        }
    }

    pub fn set_on_key_pressed_callback(&mut self, callback: F) {
        self.on_button_pressed_callback = Some(callback);
    }

    pub fn take_on_key_pressed_callback(&mut self) -> Option<F> {
        self.on_button_pressed_callback.take()
    }

    pub fn remove_on_key_pressed_callback(&mut self) {
        self.on_button_pressed_callback = None;
    }

    pub fn set_on_key_released_callback(&mut self, callback: F) {
        self.on_button_released_callback = Some(callback);
    }

    pub fn take_on_key_released_callback(&mut self) -> Option<F> {
        self.on_button_released_callback.take()
    }

    pub fn remove_on_key_released_callback(&mut self) {
        self.on_button_released_callback = None;
    }

    pub fn set_on_key_held_callback(&mut self, callback: F) {
        self.on_button_held_callback = Some(callback);
    }

    pub fn take_on_key_held_callback(&mut self) -> Option<F> {
        self.on_button_held_callback.take()
    }

    pub fn remove_on_key_held_callback(&mut self) {
        self.on_button_held_callback = None;
    }

    pub fn set_on_mouse_moved_callback(&mut self, callback: G) {
        self.on_mouse_moved_callback = Some(callback);
    }

    pub fn take_on_mouse_moved_callback(&mut self) -> Option<G> {
        self.on_mouse_moved_callback.take()
    }

    pub fn remove_on_mouse_moved_callback(&mut self) {
        self.on_mouse_moved_callback = None;
    }

    pub fn set_on_mouse_scrolled_callback(&mut self, callback: H) {
        self.on_mouse_scrolled_callback = Some(callback);
    }

    pub fn take_on_mouse_scrolled_callback(&mut self) -> Option<H> {
        self.on_mouse_scrolled_callback.take()
    }

    pub fn remove_on_mouse_scrolled_callback(&mut self) {
        self.on_mouse_scrolled_callback = None;
    }

    pub fn remove_all_callbacks(&mut self) {
        self.on_button_pressed_callback = None;
        self.on_button_released_callback = None;
        self.on_button_held_callback = None;
        self.on_mouse_moved_callback = None;
        self.on_mouse_scrolled_callback = None;
    }

    pub fn set_lock_cursor(&mut self, lock: bool) {
        self.is_cursor_locked = lock;
    }

    pub fn run(controller: Arc<RwLock<Self>>, new_main_function: fn(Arc<RwLock<Self>>, TerminatorSignalSender, TerminatorSignalReceiver)) {
        let event_loop = controller.write().unwrap().event_loop.take().unwrap();
        let mut frame_count = 0;
        let mut last_fps_print = std::time::Instant::now();

        let (render_sender, main_receiver) = mpsc::channel();
        let (main_sender, render_receiver) = mpsc::channel();

        let controller_cpy = controller.clone();

        let mut new_main_handle = Some(std::thread::spawn(move || {
            new_main_function(controller_cpy, main_sender, main_receiver);
        }));
        
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
                    WindowEvent::KeyboardInput {input, ..} => {
                        todo!("Add key press handling");
                    },
                    WindowEvent::CursorMoved {  position, .. } => {
                        todo!("Add mouse move handling");
                    },
                    WindowEvent::MouseWheel { delta, .. } => {
                        todo!("Add mouse scroll handling");
                    },
                    WindowEvent::MouseInput { button, .. } => {
                        todo!("Add mouse button handling");
                    }
                    _ => {}
                },
                Event::LoopDestroyed => {
                    close = true;
                }
                _ => {}
            }

            match render_receiver.try_recv() {
                Ok(_) => {
                    *control_flow = ControlFlow::Exit;
                },
                Err(err) => match err {
                    mpsc::TryRecvError::Empty => {},
                    mpsc::TryRecvError::Disconnected => {
                        println!("Main thread disconnected. Exiting...");
                        *control_flow = ControlFlow::Exit;
                    }
                }
                
            }

            if close {
                let _ = render_sender.send(());
                new_main_handle.take().unwrap().join().unwrap();
                {
                    let mut controller_lock = controller.write().unwrap();
                    controller_lock.vk_controller.cleanup();
                }
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

unsafe impl<F: Fn(KeyType)+'static, G: Fn(WindowPixelPosition) + 'static, H: Fn(MouseScrollDelta) + 'static> Send for ArtewaldEngine<F, G, H> {}
unsafe impl<F: Fn(KeyType)+'static, G: Fn(WindowPixelPosition) + 'static, H: Fn(MouseScrollDelta) + 'static> Sync for ArtewaldEngine<F, G, H> {}
