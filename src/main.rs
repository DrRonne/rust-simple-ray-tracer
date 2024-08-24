use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, DeviceEvent, MouseButton, StartCause, KeyEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use log::error;
use error_iter::ErrorIter as _;
use std::time::Instant;
mod engine;
use crate::engine::renderer::Renderer;
use crate::engine::camera::Camera;
use crate::engine::world::World;
use crate::engine::sphere::Sphere;
use crate::engine::cframe::Positionable;
use crate::engine::render::Renderable;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const CAMERA_MOVE_SPEED: f32 = 0.3;
const CAMERA_ROTATE_SPEED: f32 = 0.001;

fn main() -> Result<(), Error> {
    env_logger::init();
    let mut renderer = Renderer::new(WIDTH, HEIGHT);
    renderer.init().expect("Failed to initialize renderer");
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let mut now = Instant::now();
    let mut camera = Camera::new(90f32, 0.1f32);
    let mut world = World::new();
    let mut sphere = Sphere::new(10f32);
    let mut sphere2 = Sphere::new(10f32);
    let mut floor = Sphere::new(100000f32);
    sphere.set_position(-10f32, 15f32, -70f32);
    sphere2.set_position(15f32, 5f32, -70f32);
    floor.set_position(0f32, -100002f32, 0f32);
    sphere.set_color(0xffu8, 0x00u8, 0x00u8);
    sphere2.set_color(0x00u8, 0xffu8, 0x00u8);
    floor.set_color(0x00u8, 0x00u8, 0xffu8);
    world.push_renderable(Box::new(sphere));
    world.push_renderable(Box::new(sphere2));
    world.push_renderable(Box::new(floor));
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

    event_loop.run(move |event: Event<()>, event_loop| {
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
                    device_id, state, button
                } => {
                    if button == MouseButton::Left {
                        clicked = state == ElementState::Pressed;
                    }
                }
                WindowEvent::RedrawRequested => {
                    let mut movesize = (forward * forward + to_side * to_side).sqrt().max(1.0f32);
                    camera.cframe.multiply_vector(to_side / movesize * CAMERA_MOVE_SPEED, 0f32, forward / movesize * CAMERA_MOVE_SPEED);
                    if clicked {
                        camera.cframe.multiply_angles(cursor_top * CAMERA_ROTATE_SPEED, cursor_side * CAMERA_ROTATE_SPEED, 0f32);
                    }
                    let elapsed = now.elapsed();
                    println!("Elapsed: {:.2?}", elapsed);
                    now = Instant::now();
                    let render_objects = world.get_render_objects();
                    let directionlight_direction = world.get_direction_light_direction_vec();
                    let directionlight_color = world.get_direction_light_color_vec();
                    let mut vec = renderer.render_frame(camera, render_objects, directionlight_direction, directionlight_color).expect("failed to render frame");
                    let mut frame = pixels.frame_mut();
                    frame.copy_from_slice(&mut vec[..]);
                    // world.draw(pixels.frame_mut());
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
