use crate::engine::cframe::CFrame;

pub trait Renderable {
    fn get_render_object(&mut self) -> RenderObject;
    fn set_color(&mut self, red: u8, green: u8, blue: u8);
    fn set_reflectance(&mut self, reflectance: f32);
    fn set_transparency(&mut self, transparency: f32);
    fn set_refractive_index(&mut self, refractive_index: f32);
}

#[derive(Copy, Clone)]
pub enum RenderType {
    SPHERE = 0,
}

pub struct RenderObject {
    cframe: CFrame,
    render_type: RenderType,
    object_props: Vec<f32>,
    color: Vec<u8>,
    reflectance: f32,
    transparency: f32,
    refractive_index: f32,
}

impl RenderObject {
    pub fn new(cframe: CFrame, render_type: RenderType, object_props: Vec<f32>, color: Vec<u8>, reflectance: f32, transparency: f32, refractive_index: f32) -> Self {
        Self {
            cframe,
            render_type,
            object_props,
            color,
            reflectance,
            transparency,
            refractive_index,
         }
    }

    pub fn convert_to_cframe_buffer(&mut self) -> Vec<f32> {
        return self.cframe.to_vec();
    }

    pub fn get_render_type(&mut self) -> u8 {
        return self.render_type as u8;
    }

    pub fn get_object_props_vec(&mut self) -> Vec<f32> {
        return self.object_props.clone();
    }

    pub fn get_color_vec(&mut self) -> Vec<u8> {
        return self.color.clone();
    }

    pub fn get_reflectance(&mut self) -> f32 {
        return self.reflectance;
    }

    pub fn get_transparency(&mut self) -> f32 {
        return self.transparency;
    }

    pub fn get_refractive_index(&mut self) -> f32 {
        return self.refractive_index;
    }
}