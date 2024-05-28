use std::{borrow::BorrowMut, collections::{hash_map, HashMap}, ffi::CString, sync::{Arc, RwLock}, time::Instant};

use ash::vk;
use graphics_objects::{TextureResource, UniformBufferResource};
use pipeline_manager::ShaderInfo;
use test_objects::{SimpleRenderableObject, TwoDPositionSimpleRenderableObject};
use vertex::{generate_circle_type_one, generate_circle_type_three, generate_circle_type_two, SimpleVertex};
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
mod object_manager;

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().with_title("Artewald Engine 2").build(&event_loop).unwrap();

    let mut vk_controller = VkController::new(window, "Artewald Engine 2");
    let mut swapchain_extent = vk_controller.get_swapchain_extent();

    let (vertices, indices) = load_model("./assets/objects/viking_room.obj");
    
    let mod1 = glm::translate(&glm::identity(), &glm::Vec3::new(-1.5, 0.0, 0.0)) * glm::rotate(&glm::identity(), 0f32 * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0));

    let mod2 = glm::translate(&glm::identity(), &glm::Vec3::new(1.5, 0.0, 0.0)) * glm::rotate(&glm::identity(), 0f32 * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0));

    let mut proj = glm::perspective(swapchain_extent.width as f32 / swapchain_extent.height as f32, 90.0_f32.to_radians(), 0.1, 10.0);
    proj[(1, 1)] *= -1.0;
    let view_projection = Arc::new(RwLock::new(UniformBufferResource {
        buffer: proj * glm::look_at(&glm::vec3(2.0, 2.0, 2.0), &glm::vec3(0.0, 0.0, 0.0), &glm::vec3(0.0, 0.0, 1.0)),
        binding: 1,
    }));

    let texture = Arc::new(RwLock::new(TextureResource {
        image: image::open("./assets/images/viking_room.png").unwrap(),
        binding: 2,
        stage: vk::ShaderStageFlags::FRAGMENT,
    }));

    let obj1 = Arc::new(RwLock::new(SimpleRenderableObject {
        vertices: vertices.clone(),
        indices: indices.clone(),
        model_matrix: Arc::new(RwLock::new(UniformBufferResource { buffer: mod1, binding: 0 })),
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
        view_projection: view_projection.clone(),
        texture: texture.clone(),
    }));

    let obj2 = Arc::new(RwLock::new(SimpleRenderableObject {
        vertices: vertices.clone(),
        indices: indices.clone(),
        model_matrix: Arc::new(RwLock::new(UniformBufferResource { buffer: mod2, binding: 0 })),
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
        view_projection: view_projection.clone(),
        texture: texture.clone(),
    }));

    

    let object_ids = vk_controller.add_objects_to_render(vec![obj1.clone(), obj2.clone()]).unwrap();
    
    // let num_vertices = 12;//49152*4;

    // println!("1");
    // let (vertices_one, indices_one) = generate_circle_type_one(1.0, num_vertices);
    // println!("2");
    // let (vertices_two, indices_two) = generate_circle_type_two(1.0, num_vertices);
    // println!("3");
    // let start_time = Instant::now();
    // println!("Start time: {:?}", start_time.elapsed().as_secs_f32());
    // let (vertices_three, indices_three) = generate_circle_type_three(1.0, num_vertices);
    // println!("End time: {:?}", start_time.elapsed().as_secs_f32());
    // println!("4");

    // let obj_one = Arc::new(RwLock::new(TwoDPositionSimpleRenderableObject {
    //     vertices: vertices_one,
    //     indices: indices_one,
    //     shaders: vec![
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.vert"),
    //             shader_stage_flag: vk::ShaderStageFlags::VERTEX,
    //             entry_point: CString::new("main").unwrap(),
    //         },
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.frag"),
    //             shader_stage_flag: vk::ShaderStageFlags::FRAGMENT,
    //             entry_point: CString::new("main").unwrap(),
    //         }
    //     ],
    //     descriptor_set_layout: None,
    // }));

    // let obj_two = Arc::new(RwLock::new(TwoDPositionSimpleRenderableObject {
    //     vertices: vertices_two,
    //     indices: indices_two,
    //     shaders: vec![
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.vert"),
    //             shader_stage_flag: vk::ShaderStageFlags::VERTEX,
    //             entry_point: CString::new("main").unwrap(),
    //         },
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.frag"),
    //             shader_stage_flag: vk::ShaderStageFlags::FRAGMENT,
    //             entry_point: CString::new("main").unwrap(),
    //         }
    //     ],
    //     descriptor_set_layout: None,
    // }));

    // let obj_three = Arc::new(RwLock::new(TwoDPositionSimpleRenderableObject {
    //     vertices: vertices_three,
    //     indices: indices_three,
    //     shaders: vec![
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.vert"),
    //             shader_stage_flag: vk::ShaderStageFlags::VERTEX,
    //             entry_point: CString::new("main").unwrap(),
    //         },
    //         ShaderInfo {
    //             path: std::path::PathBuf::from("./assets/shaders/circle.frag"),
    //             shader_stage_flag: vk::ShaderStageFlags::FRAGMENT,
    //             entry_point: CString::new("main").unwrap(),
    //         }
    //     ],
    //     descriptor_set_layout: None,
    // }));

    // let mut current_object_id = vk_controller.add_object_to_render(obj_three.clone()).unwrap();

    let mut frame_count = 0;
    let mut last_fps_print = std::time::Instant::now();
    let start_time = Instant::now();

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
                vk_controller.cleanup();
                close = true;
            }
            _ => {}
        }

        if close {
            return;
        }
        swapchain_extent = vk_controller.get_swapchain_extent();
        
        obj1.write().unwrap().model_matrix.write().unwrap().buffer = glm::translate(&glm::identity(), &glm::Vec3::new(-1.5, 0.0, 0.0)) * glm::rotate(&glm::identity(), start_time.elapsed().as_secs_f32() * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0));
        obj2.write().unwrap().model_matrix.write().unwrap().buffer = glm::translate(&glm::identity(), &glm::Vec3::new(1.5, 0.0, 0.0)) * glm::rotate(&glm::identity(), start_time.elapsed().as_secs_f32() * std::f32::consts::PI * 0.25, &glm::vec3(0.0, 0.0, 1.0));

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
