#[derive(Copy, Clone)]
pub struct CFrame {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r00: f32,
    pub r01: f32,
    pub r02: f32,
    pub r10: f32,
    pub r11: f32,
    pub r12: f32,
    pub r20: f32,
    pub r21: f32,
    pub r22: f32,
}

impl CFrame {
    pub fn new(x: f32, y: f32, z: f32, r00: f32, r01: f32, r02: f32, r10: f32, r11: f32, r12: f32, r20: f32, r21: f32, r22: f32) -> Self {
        Self {
            x, y, z, r00, r01, r02, r10, r11, r12, r20, r21, r22,
        }
    }

    pub fn new_from_pos(x: f32, y: f32, z: f32) -> Self {
        Self {
            x, y, z, 
            r00: 1.0f32, r01: 0.0f32, r02: 0.0f32,
            r10: 0.0f32, r11: 1.0f32, r12: 0.0f32,
            r20: 0.0f32, r21: 0.0f32, r22: 1.0f32,
        }
    }

    pub fn to_vec(&mut self) -> Vec<f32> {
        return vec![self.x, self.y, self.z, self.r00, self.r01, self.r02, self.r10, self.r11, self.r12, self.r20, self.r21, self.r22];
    }

    pub fn multiply_vector(&mut self, x: f32, y: f32, z: f32) {
        self.x = self.r00 * x + self.r10 * y + self.r20 * z + self.x;
        self.y = self.r01 * x + self.r11 * y + self.r21 * z + self.y;
        self.z = self.r02 * x + self.r12 * y + self.r22 * z + self.z;
    }

    pub fn multiply_angles(&mut self, alpha: f32, beta: f32, gamma: f32) {
        let sa: f32 = alpha.sin();
        let ca: f32 = alpha.cos();
        let sb: f32 = beta.sin();
        let cb: f32 = beta.cos();
        let sg: f32 = gamma.sin();
        let cg: f32 = gamma.cos();
        let r00: f32 = cb * cg;
        let r01: f32 = sa * sb * cg - ca * sg;
        let r02: f32 = ca * sb * cg + sa * sg;
        let r10: f32 = cb * sg;
        let r11: f32 = sa * sb * sg + ca * cg;
        let r12: f32 = ca * sb * sg - sa * cg;
        let r20: f32 = -sb;
        let r21: f32 = sa * cb;
        let r22: f32 = ca * cb;

        let tr00: f32 = self.r00 * r00 + self.r01 * r10 + self.r02 * r20;
        let tr01: f32 = self.r00 * r01 + self.r01 * r11 + self.r02 * r21;
        let tr02: f32 = self.r00 * r02 + self.r01 * r12 + self.r02 * r22;
        let tr10: f32 = self.r10 * r00 + self.r11 * r10 + self.r12 * r20;
        let tr11: f32 = self.r10 * r01 + self.r11 * r11 + self.r12 * r21;
        let tr12: f32 = self.r10 * r02 + self.r11 * r12 + self.r12 * r22;
        let tr20: f32 = self.r20 * r00 + self.r21 * r10 + self.r22 * r20;
        let tr21: f32 = self.r20 * r01 + self.r21 * r11 + self.r22 * r21;
        let tr22: f32 = self.r20 * r02 + self.r21 * r12 + self.r22 * r22;

        self.r00 = tr00;
        self.r01 = tr01;
        self.r02 = tr02;
        self.r10 = tr10;
        self.r11 = tr11;
        self.r12 = tr12;
        self.r20 = tr20;
        self.r21 = tr21;
        self.r22 = tr22;
    }
}

impl Default for CFrame {
    fn default() -> CFrame {
        CFrame {
            x: 0.0f32, y: 0.0f32, z: 0.0f32,
            r00: 1.0f32, r01: 0.0f32, r02: 0.0f32,
            r10: 0.0f32, r11: 1.0f32, r12: 0.0f32,
            r20: 0.0f32, r21: 0.0f32, r22: 1.0f32,
        }
    }
}

pub trait Positionable {
    fn set_cframe(&mut self, cframe: CFrame);
    fn set_position(&mut self, x: f32, y: f32, z: f32);
}