pub struct DirectionLight {
    direction: [f32; 3],
    color: [u8; 3],
}

impl DirectionLight {
    pub fn new(direction: [f32; 3], color: [u8; 3]) -> Self {
        Self {
            direction,
            color,
         }
    }

    pub fn get_direction(&self) -> [f32; 3] {
        self.direction
    }

    pub fn get_color(&self) -> [u8; 3] {
        self.color
    }

    pub fn set_direction(&mut self, direction: [f32; 3]) {
        let len = (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]).sqrt();
        if len != 0.0 {
            self.direction = [direction[0] / len, direction[1] / len, direction[2] / len];
        } else {
            self.direction = [1.0f32, 0.0f32, 0.0f32]; // Default to a direction along the x-axis if the input is zero
        }
    }

    pub fn set_color(&mut self, color: [u8; 3]) {
        self.color = color;
    }
}

// By default, the directional light is a white light, 45° in every direction
impl Default for DirectionLight {
    fn default() -> DirectionLight {
        DirectionLight {
            // direction: vec![0.0f32, -1.0f32, 0.0f32],
            // direction: vec![0.70710678118f32, -0.70710678118f32, 0f32],
            direction: [0.577350269f32, -0.577350269f32, -0.577350269f32],
            color: [0xffu8, 0xffu8, 0xffu8],
        }
    }
}