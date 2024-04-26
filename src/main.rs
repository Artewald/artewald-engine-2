use std::{collections::{hash_map, HashMap}, ffi::CString, sync::{Arc, RwLock}};

use ash::vk;
use graphics_objects::{TextureResource, UniformBufferObject, UniformBufferResource};
use pipeline_manager::ShaderInfo;
use test_objects::SimpleRenderableObject;
use vertex::SimpleVertex;
use vk_controller::{VkController, VkControllerGraphicsObjectsControl};
use winit::{event_loop::{EventLoop, ControlFlow}, window::WindowBuilder, event::{Event, WindowEvent, ElementState, KeyboardInput}};
use nalgebra_glm as glm;

mod vk_controller;
mod vertex;
mod graphics_objects;
mod vk_allocator;
mod pipeline_manager;
mod sampler_manager;
mod test_objects;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Artewald Engine 2").build(&event_loop).unwrap();

    let mut vk_controller = VkController::new(window, "Artewald Engine 2");
    let mut swapchain_extent = vk_controller.get_swapchain_extent();

    let (vertices, indices) = load_model("./assets/objects/viking_room.obj");
    
    let mut ubo = UniformBufferObject {
        model: glm::rotate(&glm::identity(), 0f32 * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0)),
        view: glm::look_at(&glm::vec3(2.0, 2.0, 2.0), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 0.0, 1.0)),
        proj: glm::perspective(swapchain_extent.width as f32 / swapchain_extent.height as f32, 90.0_f32.to_radians(), 0.1, 10.0),
    };
    ubo.proj[(1, 1)] *= -1.0;

    let obj = Arc::new(RwLock::new(SimpleRenderableObject {
        vertices,
        indices,
        uniform_buffer: Arc::new(RwLock::new(UniformBufferResource { buffer: ubo, binding: 0 })),
        texture: Arc::new(RwLock::new(TextureResource {
            image: image::open("./assets/images/viking_room.png").unwrap(),
            binding: 1,
            stage: vk::ShaderStageFlags::FRAGMENT,
        })),
        shaders: vec![
            ShaderInfo {
                path: std::path::PathBuf::from("./assets/shaders/triangle.vert"),
                shader_stage_flag: vk::ShaderStageFlags::VERTEX,
                entry_point: CString::new("main").unwrap(),
            },
            ShaderInfo {
                path: std::path::PathBuf::from("./assets/shaders/triangle.frag"),
                shader_stage_flag: vk::ShaderStageFlags::FRAGMENT,
                entry_point: CString::new("main").unwrap(),
            }
        ],
        descriptor_set_layout: None,
    }));
    
    vk_controller.add_object_to_render(obj.clone()).unwrap();

    let mut frame_count = 0;
    let mut last_fps_print = std::time::Instant::now();
    let start_time = std::time::Instant::now();

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
        swapchain_extent = vk_controller.get_swapchain_extent();
        
        let mut ubo = UniformBufferObject {
            model: glm::rotate(&glm::identity(), start_time.elapsed().as_secs_f32() * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0)),
            view: glm::look_at(&glm::vec3(2.0, 2.0, 2.0), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 0.0, 1.0)),
            proj: glm::perspective(swapchain_extent.width as f32 / swapchain_extent.height as f32, 90.0_f32.to_radians(), 0.1, 10.0),
        };
        ubo.proj[(1, 1)] *= -1.0;
        {
            let obj_locked = obj.write().unwrap();
            let mut ubo_locked = obj_locked.uniform_buffer.write().unwrap();
            ubo_locked.buffer = ubo;
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

fn load_model(path: &str) -> (Vec<SimpleVertex>, Vec<u32>) {
    let (models, _) = tobj::load_obj(path, &tobj::LoadOptions::default()).unwrap();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut unique_vertices: HashMap<SimpleVertex, u32> = HashMap::new();

    for model in models {
        let mesh = model.mesh;
        for i in 0..mesh.indices.len() {
            let index = mesh.indices[i] as usize;
            let vertex = SimpleVertex {
                position: glm::vec3(mesh.positions[index * 3], mesh.positions[index * 3 + 1], mesh.positions[index * 3 + 2]),
                color: glm::vec3(1.0, 1.0, 1.0),
                tex_coord: glm::vec2(mesh.texcoords[index * 2], 1.0 - mesh.texcoords[index * 2 + 1]),
            };
    
            if let hash_map::Entry::Vacant(e) = unique_vertices.entry(vertex) {
                e.insert(vertices.len() as u32);
                vertices.push(vertex);
            }
            indices.push(*unique_vertices.get(&vertex).unwrap());
        }
    }

    // vertices = TEST_RECTANGLE.to_vec();
    // indices = TEST_RECTANGLE_INDICES.to_vec();

    (vertices, indices)
}
