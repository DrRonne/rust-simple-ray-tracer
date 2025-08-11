use crate::engine::util::cframe::{CFrame, Positionable};
use crate::engine::render::{Renderable, RenderObject, RenderType};

pub struct Sphere {
    cframe: CFrame,
    radius: f32,
    color: Vec<u8>,
    reflectance: f32,
    transparency: f32,
    refractive_index: f32,
}

impl Sphere {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            ..Default::default()
         }
    }
}

impl Default for Sphere {
    fn default() -> Self {
        Self {
            cframe: CFrame::default(),
            radius: 0.0,
            color: vec![0, 0, 0],
            reflectance: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
        }
    }
}

impl Renderable for Sphere {
    fn get_render_object(&mut self) -> RenderObject {
        return RenderObject::new(self.cframe, RenderType::SPHERE, vec![self.radius], self.color.clone(), self.reflectance, self.transparency, self.refractive_index);
    }

    fn set_color(&mut self, red: u8, green: u8, blue: u8) {
        self.color = vec![red, green, blue];
    }

    fn set_reflectance(&mut self, reflectance: f32) {
        self.reflectance = reflectance;
    }

    fn set_transparency(&mut self, transparency: f32) {
        self.transparency = transparency;
    }

    fn set_refractive_index(&mut self, refractive_index: f32) {
        self.refractive_index = refractive_index;
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