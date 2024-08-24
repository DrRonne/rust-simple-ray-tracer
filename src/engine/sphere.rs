use crate::engine::cframe::{CFrame, Positionable};
use crate::engine::render::{Renderable, RenderObject, RenderType};

#[derive(Default)]
pub struct Sphere {
    cframe: CFrame,
    radius: f32,
    color: Vec<u8>,
}

impl Sphere {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            ..Default::default()
         }
    }
}

impl Renderable for Sphere {
    fn get_render_object(&mut self) -> RenderObject {
        return RenderObject::new(self.cframe, RenderType::SPHERE, vec![self.radius], self.color.clone());
    }

    fn set_color(&mut self, red: u8, green: u8, blue: u8) {
        self.color = vec![red, green, blue];
    }
}

impl Positionable for Sphere {
    fn set_cframe(&mut self, cframe: CFrame) {
        self.cframe = cframe;
    }

    fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.cframe = CFrame::new(x, y, z, self.cframe.r00, self.cframe.r01, self.cframe.r02, self.cframe.r10, self.cframe.r11, self.cframe.r12, self.cframe.r20, self.cframe.r21, self.cframe.r22);
    }
}