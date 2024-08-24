use crate::engine::cframe::CFrame;

#[derive(Default, Copy, Clone)]
pub struct Camera {
    fov: f32,
    focal_length: f32,
    pub cframe: CFrame,
}

impl Camera {
    pub fn new(fov: f32, focal_length: f32) -> Self {
        Self {
            fov,
            focal_length,
            ..Default::default()
         }
    }

    pub fn to_vec(&mut self) -> Vec<f32> {
        let mut vec = self.cframe.to_vec();
        vec.push(self.fov);
        return vec;
    }

    pub fn get_fov(&mut self) -> f32 {
        return self.fov;
    }

    pub fn get_focal_length(&mut self) -> f32 {
        return self.focal_length;
    }
}