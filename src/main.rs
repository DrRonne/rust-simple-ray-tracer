use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, DeviceEvent, MouseButton, KeyEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use log::error;
use error_iter::ErrorIter as _;
use std::time::Instant;
mod engine;
use crate::engine::render::renderer::Renderer;
use crate::engine::camera::Camera;
use crate::engine::world::World;
use crate::engine::primitives::primitive::Primitive;
use crate::engine::util::cframe::Positionable;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const CAMERA_MOVE_SPEED: f32 = 0.3;
const CAMERA_ROTATE_SPEED: f32 = 0.001;

fn main() -> Result<(), Error> {
    env_logger::init();
    let mut renderer = Renderer::new(WIDTH, HEIGHT);
    renderer.init().expect("Failed to initialize renderer");
    let event_loop = EventLoop::new().unwrap();
    let mut now = Instant::now();
    let mut camera = Camera::new(90f32, 0.1f32);
    let mut world = World::new();
    let mut sphere = Primitive::new_sphere(10f32);
    let mut sphere2 = Primitive::new_sphere(10f32);
    let mut sphere3 = Primitive::new_sphere(10f32);
    let mut sphere4 = Primitive::new_sphere(12f32);
    let mut sphere5 = Primitive::new_sphere(10f32);
    let mut sphere6 = Primitive::new_sphere(5f32);
    let mut sphere7 = Primitive::new_sphere(5f32);
    let mut sphere8 = Primitive::new_sphere(5f32);
    let mut floor = Primitive::new_sphere(30f32);
    let mut floor2 = Primitive::new_sphere(30f32);
    sphere.set_position(0f32, 0f32, -40f32);
    sphere2.set_position(0f32, 0f32, 40f32);
    sphere3.set_position(-20f32, 10f32, 0f32);
    sphere4.set_position(20f32, 10f32, 0f32);
    sphere5.set_position(30f32, 20f32, 10f32);
    sphere6.set_position(-35f32, 25f32, 10f32);
    sphere7.set_position(-20f32, 20f32, 25f32);
    sphere8.set_position(5f32, 10f32, 0f32);
    floor.set_position(10f32, -32f32, -5f32);
    floor2.set_position(40f32, -32f32, -25f32);
    sphere.set_color([0xffu8, 0x00u8, 0x00u8]);
    sphere.set_reflectance(0.9f32);
    sphere2.set_color([0x00u8, 0xffu8, 0x00u8]);
    sphere2.set_reflectance(0.0f32);
    sphere2.set_transparency(1.0f32);
    sphere2.set_refractive_index(1.05f32);
    sphere3.set_color([0xffu8, 0x00u8, 0xffu8]);
    sphere3.set_transparency(0.9f32);
    sphere3.set_refractive_index(1.0f32);
    sphere4.set_color([0x00u8, 0xffu8, 0xffu8]);
    sphere4.set_render_radius(15f32);
    sphere5.set_render_radius(15f32);
    sphere8.set_render_radius(8f32);
    sphere8.set_color([0x00u8, 0x00u8, 0xffu8]);
    sphere8.set_reflectance(0.5f32);
    // sphere4.set_transparency(0.5f32);
    sphere5.set_color([0xffu8, 0xffu8, 0x00u8]);
    sphere5.set_transparency(0.5f32);
    sphere5.set_refractive_index(1.05f32);
    sphere6.set_color([0x80u8, 0xffu8, 0x80u8]);
    sphere7.set_color([0xffu8, 0x80u8, 0x80u8]);
    sphere7.set_transparency(0.5f32);
    sphere7.set_refractive_index(1.0f32);
    // floor.set_reflectance(0.0f32);
    floor.set_color([0xffu8, 0xffu8, 0xffu8]);
    floor2.set_color([0xffu8, 0xffu8, 0xffu8]);
    let s4 = world.push_primitive(sphere4);
    let s5 = world.push_primitive(sphere5);
    let s8 = world.push_primitive(sphere8);
    world.merge_primitives(s4, s5, 5f32);
    world.merge_primitives(s4, s8, 1f32);
    world.push_primitive(sphere);
    world.push_primitive(sphere2);
    world.push_primitive(sphere3);
    world.push_primitive(sphere6);
    world.push_primitive(sphere7);
    world.push_primitive(floor);
    world.push_primitive(floor2);
    // for i in 0..1000 {
    //     let mut s = Sphere::new(10f32);
    //     s.set_color(0xff, 0xff, 0x00);
    //     s.set_position(i as f32 * 10f32, 0f32, -50f32);
    //     world.push_renderable(Box::new(s));
    // }
    let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
    let window = WindowBuilder::new()
        .with_title("Simple ray tracer")
        .with_inner_size(size)
        .with_min_inner_size(size)
        .build(&event_loop)
        .unwrap();

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH.into(), HEIGHT.into(), surface_texture)?
    };

    let mut forward = 0f32;
    let mut to_side = 0f32;
    let mut clicked = false;
    let mut cursor_side = 0f32;
    let mut cursor_top = 0f32;
    let mut yaw = 0f32;
    let mut pitch = 0f32;

    let _ = event_loop.run(move |event: Event<()>, event_loop| {
        // Draw the current frame
        if let Event::DeviceEvent { ref event, .. } = event {
            match event {
                DeviceEvent::MouseMotion {
                    delta
                } => {
                    cursor_side = delta.0 as f32;
                    cursor_top = delta.1 as f32;
                }
                _ => (),
            }
        }
        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyW),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    forward = -1f32;
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyW),
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    if forward < 0f32 {
                        forward = 0f32;
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyS),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    forward = 1f32;
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyS),
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    if forward > 0f32 {
                        forward = 0f32;
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyA),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    to_side = -1f32;
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyA),
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    if to_side < 0f32 {
                        to_side = 0f32;
                    }
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyD),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    to_side = 1f32;
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::KeyD),
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => {
                    if to_side > 0f32 {
                        to_side = 0f32;
                    }
                }
                WindowEvent::MouseInput {
                    device_id: _, state, button
                } => {
                    if button == MouseButton::Left {
                        clicked = state == ElementState::Pressed;
                    }
                }
                WindowEvent::RedrawRequested => {
                    let movesize = (forward * forward + to_side * to_side).sqrt().max(1.0f32);
                    if clicked {
                        camera.reset_rotation();
                        yaw += cursor_side * CAMERA_ROTATE_SPEED;
                        pitch += cursor_top * CAMERA_ROTATE_SPEED;
                        camera.cframe.multiply_angles(pitch, 0f32, 0f32);
                        camera.cframe.multiply_angles(0f32, yaw, 0f32);
                        cursor_side = 0f32;
                        cursor_top = 0f32;
                    }
                    camera.cframe.multiply_vector(to_side / movesize * CAMERA_MOVE_SPEED, 0f32, forward / movesize * CAMERA_MOVE_SPEED);
                    let elapsed = now.elapsed();
                    println!("Elapsed: {:.2?}", elapsed);
                    now = Instant::now();
                    let directionlight_direction = world.get_direction_light_direction_vec();
                    let directionlight_color = world.get_direction_light_color_vec();
                    let primitives = world.get_primitives();
                    let mut vec = renderer.render_frame(camera, directionlight_direction, directionlight_color, primitives, world.get_octree_nodes(), world.get_octree_root_node_size(), world.get_octree_root_position(), world.get_octree_root_node_indices(), world.get_octree_dimensions()).expect("failed to render frame");
                    let frame = pixels.frame_mut();
                    frame.copy_from_slice(&mut vec[..]);
                    if let Err(err) = pixels.render() {
                        log_error("pixels.render", err);
                        event_loop.exit();
                        return;
                    }
                    window.request_redraw();
                }
                _ => (),
            }
        }
    });

    Ok(())
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}
