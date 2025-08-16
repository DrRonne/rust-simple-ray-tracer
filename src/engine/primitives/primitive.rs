use std::fmt;
use ocl::OclPrm;

use crate::engine::primitives::sphere::SphereData;
use crate::engine::util::cframe::{CFrame, Positionable};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    SPHERE = 0,
}

impl Default for Kind {
    fn default() -> Self {
        Kind::SPHERE
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union PrimitivePayload {
    pub sphere_data: SphereData,
    // pub plane_data: PlaneData, // Add new payloads for primitives here
    pub raw: [f32; 1], // raw fallback, make sure to update this and all padding of the structs when you add a new type of primitive!
}

impl Default for PrimitivePayload {
    fn default() -> Self {
        PrimitivePayload {
            raw: [0.0; 1]
        }
    }
}

impl PartialEq for PrimitivePayload {
    fn eq(&self, other: &Self) -> bool {
        unsafe { self.raw == other.raw }
    }
}

impl fmt::Debug for PrimitivePayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            f.debug_tuple("PrimitivePayload")
                .field(&self.raw)
                .finish()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Primitive {
    kind: Kind,
    transparency: f32,
    refractive_index: f32,
    reflectance: f32,
    color: [u8; 3],
    cframe: CFrame,
    _pad_common: f32,

    // Type-specific fields
    pub payload: PrimitivePayload,
}

impl Primitive {
    pub fn new_sphere(radius: f32) -> Self {
        Self {
            kind: Kind::SPHERE,
            transparency: 0.0f32,
            refractive_index: 1.0f32,
            reflectance: 0.0f32,
            color: [0xFF, 0xFF, 0xFF],
            cframe: CFrame::default(),
            _pad_common: 0.0,
            payload: PrimitivePayload { sphere_data: SphereData::new(radius) },
        }
    }

    pub fn set_color(&mut self, color: [u8; 3]) {
        self.color = color;
    }

    pub fn set_transparency(&mut self, transparency: f32) {
        self.transparency = transparency;
    }

    pub fn set_refractive_index(&mut self, refractive_index: f32) {
        self.refractive_index = refractive_index;
    }

    pub fn set_reflectance(&mut self, reflectance: f32) {
        self.reflectance = reflectance;
    }
}

impl Positionable for Primitive {
    fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.cframe = CFrame::new(x, y, z, self.cframe.r00, self.cframe.r01, self.cframe.r02, self.cframe.r10, self.cframe.r11, self.cframe.r12, self.cframe.r20, self.cframe.r21, self.cframe.r22);
    }

    fn set_cframe(&mut self, cframe: CFrame) {
        self.cframe = cframe;
    }
}

// SAFETY: OctreeNode is plain-old-data (POD) and contains only OclPrm-compatible fields.
unsafe impl OclPrm for Primitive {}