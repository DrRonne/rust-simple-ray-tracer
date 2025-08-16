#[repr(C)]
#[derive(Clone, Copy)]
pub struct SphereData {
    pub radius: f32,
    pub _pad: [f32; 0], // pad so size matches largest variant's payload
}

impl SphereData {
    pub fn new(radius: f32) -> Self {
        Self {
            radius,
            _pad: [],
        }
    }
}
