use crate::engine::cframe::CFrame;

pub trait Renderable {
    fn get_render_object(&mut self) -> RenderObject;
    fn set_color(&mut self, red: u8, green: u8, blue: u8);
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
}

impl RenderObject {
    pub fn new(cframe: CFrame, render_type: RenderType, object_props: Vec<f32>, color: Vec<u8>) -> Self {
        Self {
            cframe,
            render_type,
            object_props,
            color,
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
}