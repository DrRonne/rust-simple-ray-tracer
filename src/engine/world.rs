use crate::engine::render::{Renderable, RenderObject};
use crate::engine::lights::directionlight::DirectionLight;

#[derive(Default)]
pub struct World {
    objects: Vec<Box<dyn Renderable>>,
    directionlight: DirectionLight,
}

impl World {
    pub fn new() -> Self {
        Self {
            ..Default::default()
         }
    }

    pub fn get_render_objects(&mut self) -> Vec<RenderObject> {
        let mut render_objects: Vec<RenderObject> = Vec::new();
        for obj in self.objects.iter_mut() {
            render_objects.push(obj.get_render_object());
        }
        return render_objects;
    }

    pub fn push_renderable(&mut self, render_object: Box<dyn Renderable>) {
        self.objects.push(render_object);
    }

    pub fn get_direction_light_direction_vec(&mut self) -> Vec<f32> {
        return self.directionlight.get_direction();
    }

    pub fn get_direction_light_color_vec(&mut self) -> Vec<u8> {
        return self.directionlight.get_color();
    }
}