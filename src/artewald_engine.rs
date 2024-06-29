use std::{borrow::Cow, sync::{mpsc, Arc, RwLock}};

use winit::{event::{Event, WindowEvent}, event_loop::{ControlFlow, EventLoop}, window::WindowBuilder};

use crate::{graphics_objects::GraphicsObject, inputs::{KeyType, KeyboardKeyCodes, MouseScrollDelta, WindowPixelPosition}, pipeline_manager::Vertex, vk_controller::{ObjectID, VkController, VkControllerGraphicsObjectsControl}};

pub type TerminatorSignalSender = mpsc::Sender<()>;
pub type TerminatorSignalReceiver = mpsc::Receiver<()>;

pub struct ArtewaldEngine {
    vk_controller: VkController,
    event_loop: Option<EventLoop<()>>,
    currently_pressed_keys: Vec<KeyType>,
    on_button_pressed_callback: Option<Box<dyn Fn(KeyType)>>,
    on_button_released_callback: Option<Box<dyn Fn(KeyType)>>,
    on_mouse_moved_callback: Option<Box<dyn Fn(WindowPixelPosition)>>,
    on_mouse_scrolled_callback: Option<Box<dyn Fn(MouseScrollDelta)>>,
    is_cursor_locked: bool,
}

impl ArtewaldEngine {
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
            on_mouse_moved_callback: None,
            on_mouse_scrolled_callback: None,
            is_cursor_locked: false,
        }
    }

    pub fn set_on_key_pressed_callback(&mut self, callback: Box<dyn Fn(KeyType)>) {
        self.on_button_pressed_callback = Some(callback);
    }

    pub fn take_on_key_pressed_callback(&mut self) -> Option<Box<dyn Fn(KeyType)>> {
        self.on_button_pressed_callback.take()
    }

    pub fn remove_on_key_pressed_callback(&mut self) {
        self.on_button_pressed_callback = None;
    }

    pub fn set_on_key_released_callback(&mut self, callback: Box<dyn Fn(KeyType)>) {
        self.on_button_released_callback = Some(callback);
    }

    pub fn take_on_key_released_callback(&mut self) -> Option<Box<dyn Fn(KeyType)>> {
        self.on_button_released_callback.take()
    }

    pub fn remove_on_key_released_callback(&mut self) {
        self.on_button_released_callback = None;
    }

    pub fn set_on_mouse_moved_callback(&mut self, callback: Box<dyn Fn(WindowPixelPosition)>) {
        self.on_mouse_moved_callback = Some(callback);
    }

    pub fn take_on_mouse_moved_callback(&mut self) -> Option<Box<dyn Fn(WindowPixelPosition)>> {
        self.on_mouse_moved_callback.take()
    }

    pub fn remove_on_mouse_moved_callback(&mut self) {
        self.on_mouse_moved_callback = None;
    }

    pub fn set_on_mouse_scrolled_callback(&mut self, callback: Box<dyn Fn(MouseScrollDelta)>) {
        self.on_mouse_scrolled_callback = Some(callback);
    }

    pub fn take_on_mouse_scrolled_callback(&mut self) -> Option<Box<dyn Fn(MouseScrollDelta)>> {
        self.on_mouse_scrolled_callback.take()
    }

    pub fn remove_on_mouse_scrolled_callback(&mut self) {
        self.on_mouse_scrolled_callback = None;
    }

    pub fn remove_all_callbacks(&mut self) {
        self.on_button_pressed_callback = None;
        self.on_button_released_callback = None;
        self.on_mouse_moved_callback = None;
        self.on_mouse_scrolled_callback = None;
    }

    pub fn set_lock_cursor(&mut self, lock: bool) {
        self.is_cursor_locked = lock;
    }

    fn handle_key_press(&mut self, key: KeyType) {
        if !self.currently_pressed_keys.contains(&key) {
            self.currently_pressed_keys.push(key);
            if let Some(callback) = &self.on_button_pressed_callback {
                callback(key);
            }
        }
    }

    fn handle_key_release(&mut self, key: KeyType) {
        self.currently_pressed_keys.retain(|&x| x != key);
        if let Some(callback) = &self.on_button_released_callback {
            callback(key);
        }
    }

    pub fn borrow_currently_pressed_keys(&self) -> &[KeyType] {
        &self.currently_pressed_keys
    }

    fn handle_mouse_move(&mut self, position: WindowPixelPosition) {
        if let Some(callback) = &self.on_mouse_moved_callback {
            callback(position);
        }
    }

    fn handle_mouse_scroll(&mut self, delta: MouseScrollDelta) {
        if let Some(callback) = &self.on_mouse_scrolled_callback {
            callback(delta);
        }
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
                        if let Some(key) = input.virtual_keycode {
                            match input.state {
                                winit::event::ElementState::Pressed => {
                                    {
                                        let mut controller_lock = controller.write().unwrap();
                                        controller_lock.handle_key_press(KeyType::Keyboard(key.into()));
                                    }
                                },
                                winit::event::ElementState::Released => {
                                    {
                                        let mut controller_lock = controller.write().unwrap();
                                        controller_lock.handle_key_release(KeyType::Keyboard(key.into()));
                                    }
                                }
                            }
                        }
                    },
                    WindowEvent::CursorMoved {  position, .. } => {
                        {
                            let mut controller_lock = controller.write().unwrap();
                            controller_lock.handle_mouse_move(WindowPixelPosition::new(position.x as i32, position.y as i32));
                        }
                    },
                    WindowEvent::MouseWheel { delta, .. } => {
                        {
                            let mut controller_lock = controller.write().unwrap();
                            controller_lock.handle_mouse_scroll(delta.into());
                        }
                    },
                    WindowEvent::MouseInput { button, state, .. } => {
                        {
                            let mut controller_lock = controller.write().unwrap();
                            match state {
                                winit::event::ElementState::Pressed => controller_lock.handle_key_press(KeyType::MouseButton(button.into())),
                                winit::event::ElementState::Released => controller_lock.handle_key_release(KeyType::MouseButton(button.into())),
                            }
                        
                        }
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

pub trait ArtewaldEngineGraphicsObjectsControl<T: Vertex + Clone> {
    fn add_objects_to_render(&mut self, original_objects: Vec<Arc<RwLock<dyn GraphicsObject<T>>>>) -> Result<Vec<(ObjectID, Arc<RwLock<dyn GraphicsObject<T>>>)>, Cow<'static, str>>;
    fn remove_objects_from_renderer(&mut self, object_ids: Vec<ObjectID>) -> Result<(), Cow<'static, str>>;
}

impl<T: Vertex + Clone> ArtewaldEngineGraphicsObjectsControl<T> for ArtewaldEngine {
    fn add_objects_to_render(&mut self, original_objects: Vec<Arc<RwLock<dyn GraphicsObject<T>>>>) -> Result<Vec<(ObjectID, Arc<RwLock<dyn GraphicsObject<T>>>)>, Cow<'static, str>> {
        self.vk_controller.add_objects_to_render(original_objects)
    }
    
    fn remove_objects_from_renderer(&mut self, object_ids: Vec<ObjectID>) -> Result<(), Cow<'static, str>> {
        self.vk_controller.remove_objects_to_render(object_ids)
    }
}

unsafe impl Send for ArtewaldEngine {}
unsafe impl Sync for ArtewaldEngine {}
