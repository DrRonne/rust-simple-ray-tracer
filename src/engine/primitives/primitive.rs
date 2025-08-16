use std::fmt;
use ocl::OclPrm;

use crate::engine::primitives::sphere::SphereData;
use crate::engine::util::cframe::{CFrame, Positionable};

const MAX_MERGES: usize = 6;

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
    merge_indices: [u32; MAX_MERGES],
    merge_radii: [f32; MAX_MERGES],
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
            render_radius: radius,
            merge_indices: [0, 0, 0, 0, 0, 0],
            merge_radii: [0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32],
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

    pub fn set_render_radius(&mut self, render_radius: f32) {
        self.render_radius = render_radius;
    }

    pub fn set_radius(&mut self, radius: f32) {
        if (self.kind != Kind::SPHERE) {
            println!("WARNING: Attempted to set radius on a primitive that is not a sphere!");
        } else {
            // SAFETY: if statement checked for correct structure in the union
            unsafe {
                self.payload.sphere_data.radius = radius;
            }
        }
    }

    pub fn add_merge(&mut self, merge_index: u32, merge_radius: f32) {
        if merge_radius < 0.01f32 {
            println!("WARNING: Attempted to add a merge with a radius smaller than 0.01!");
            return;
        }
        for i in 0..MAX_MERGES {
            if self.merge_radii[i] < 0.01f32 {
                self.merge_radii[i] = merge_radius;
                self.merge_indices[i] = merge_index;
                println!("merge indices: {:?}", self.merge_indices);
                println!("merge radii: {:?}", self.merge_radii);
                return;
            }
        }
        println!("WARNING: Attempted to add more merges than the maximum allowed merges!");
    }

    fn shift_used_indices_left(&mut self) {
        // For further effiency, we like to keep all used indices in the first elements
        // This could be compared to bubble sort, which is rather inefficient, but for now I predict that this function will not be used that often anyway.
        // PERFORMANCE IMPROVEMENT POSSIBLE
        for _ in 0..MAX_MERGES {
            for j in 0..MAX_MERGES-1 {
                if self.merge_radii[j] <= 0.01f32 && self.merge_radii[j + 1] > 0.01f32 {
                    self.merge_radii[j] = self.merge_radii[j + 1];
                    self.merge_indices[j] = self.merge_indices[j + 1];
                    self.merge_radii[j + 1] = 0.0f32;
                    self.merge_indices[j + 1] = 0;
                }
            }
        }
    }

    pub fn remove_merge(&mut self, merge_index: u32) {
        let mut removed = false;
        for i in 0..MAX_MERGES {
            if self.merge_indices[i] == merge_index {
                self.merge_radii[i] = 0.0f32;
                self.merge_indices[i] = 0;
                removed = true;
                break;
            }
        }
        if !removed {
            println!("WARNING: Attempted to remove merge that did not exist!");
        } else {
            self.shift_used_indices_left();
        }
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