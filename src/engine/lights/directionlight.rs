pub struct DirectionLight {
    direction: Vec<f32>,
    color: Vec<u8>,
}

impl DirectionLight {
    pub fn new(direction: Vec<f32>, color: Vec<u8>) -> Self {
        Self {
            direction,
            color,
         }
    }

    pub fn get_direction(&mut self) -> Vec<f32> {
        return self.direction.clone();
    }

    pub fn get_color(&mut self) -> Vec<u8> {
        return self.color.clone();
    }
}

// By default, the directional light is a white light, 45° in every direction
impl Default for DirectionLight {
    fn default() -> DirectionLight {
        DirectionLight {
            // direction: vec![0.0f32, -1.0f32, 0.0f32],
            // direction: vec![0.70710678118f32, -0.70710678118f32, 0f32],
            direction: vec![0.577350269f32, -0.577350269f32, -0.577350269f32],
            color: vec![0xffu8, 0xffu8, 0xffu8],
        }
    }
}