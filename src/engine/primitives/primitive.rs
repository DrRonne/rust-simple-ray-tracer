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
    render_radius: f32,
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

    pub fn get_color(&self) -> [u8; 3] {
        self.color
    }

    pub fn set_transparency(&mut self, transparency: f32) {
        self.transparency = transparency;
    }

    pub fn get_transparency(&self) -> f32 {
        self.transparency
    }

    pub fn set_refractive_index(&mut self, refractive_index: f32) {
        self.refractive_index = refractive_index;
    }

    pub fn get_refractive_index(&self) -> f32 {
        self.refractive_index
    }

    pub fn set_reflectance(&mut self, reflectance: f32) {
        self.reflectance = reflectance;
    }

    pub fn get_reflectance(&self) -> f32 {
        self.reflectance
    }

    // The render radius should never be set to something smaller than the maximum span an object can have!
    // Render radius basically means that a ray needs to be within that range of the object for it to continue checking intersections with it
    // If the render radius is set smaller than the maximum span of the object, the ray could exclude it from its calculations while it really should be excluded!
    pub fn set_render_radius(&mut self, render_radius: f32) {
        self.render_radius = render_radius;
    }

    pub fn get_render_radius(&self) -> f32 {
        self.render_radius
    }

    pub fn set_radius(&mut self, radius: f32) {
        if self.kind != Kind::SPHERE {
            println!("WARNING: Attempted to set radius on a primitive that is not a sphere!");
        } else {
            // SAFETY: if statement checked for correct structure in the union
            unsafe {
                self.payload.sphere_data.radius = radius;
            }
        }
    }

    pub fn get_radius(&self) -> f32 {
        if self.kind != Kind::SPHERE {
            println!("WARNING: Attempted to get radius on a primitive that is not a sphere!");
            0.0f32
        } else {
            // SAFETY: if statement checked for correct structure in the union
            unsafe {
                self.payload.sphere_data.radius
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

    fn get_position(&self) -> (f32, f32, f32) {
        self.cframe.get_position()
    }

    fn get_cframe(&self) -> CFrame {
        self.cframe
    }
}

// SAFETY: OctreeNode is plain-old-data (POD) and contains only OclPrm-compatible fields.
unsafe impl OclPrm for Primitive {}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT_RADIUS: f32 = 10.0f32;

    #[test]
    fn test_create_sphere_primitive() {
        let s = Primitive::new_sphere(DEFAULT_RADIUS);
        assert_eq!(s.kind, Kind::SPHERE);
        assert_eq!(s.transparency, 0.0f32);
        assert_eq!(s.reflectance, 0.0f32);
        assert_eq!(s.refractive_index, 1.0f32);
        assert_eq!(s.color[0], 0xFF);
        assert_eq!(s.color[1], 0xFF);
        assert_eq!(s.color[2], 0xFF);
        assert_eq!(s.cframe, CFrame::default());
        assert_eq!(s.render_radius, DEFAULT_RADIUS);
        assert_eq!(s.merge_indices, [0, 0, 0, 0, 0, 0]);
        assert_eq!(s.merge_radii, [0f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
        // SAFETY: just created the primitive as a sphere, so sphere_data should be the correct payload
        unsafe {
            assert_eq!(s.payload.sphere_data.radius, DEFAULT_RADIUS);
        }
    }

    #[test]
    fn test_set_get_color() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_color([0x00, 0xFF, 0xFA]);
        assert_eq!(s.get_color(), [0x00, 0xFF, 0xFA]);
    }

    #[test]
    fn test_set_get_transparency() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_transparency(0.5f32);
        assert_eq!(s.get_transparency(), 0.5f32);
    }

    #[test]
    fn test_set_get_reflectance() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_reflectance(0.5f32);
        assert_eq!(s.get_reflectance(), 0.5f32);
    }

    #[test]
    fn test_set_get_refractive_index() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_refractive_index(0.5f32);
        assert_eq!(s.get_refractive_index(), 0.5f32);
    }

    #[test]
    fn test_set_get_render_radius() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_render_radius(0.5f32);
        assert_eq!(s.get_render_radius(), 0.5f32);
    }

    #[test]
    fn test_set_get_position() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        let position = (1f32, 2f32, 3f32);
        s.set_position(position.0, position.1, position.2);
        assert_eq!(s.get_position(), position);
    }

    #[test]
    fn test_set_get_cframe() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        let angles = (0.1f32, 0.2f32, 0.3f32);
        let vector = (10f32, 20f32, 30f32);
        let mut res = CFrame::default();
        res.multiply_angles(angles.0, angles.1, angles.2);
        res.multiply_vector(vector.0, vector.1, vector.2);
        s.set_cframe(res);
        assert_eq!(s.get_cframe(), res);
    }

    #[test]
    fn test_add_merge() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        for i in 0..MAX_MERGES {
            let added_index = i as u32;
            let added_radius = (i + 1) as f32;
            s.add_merge(added_index, added_radius);
            assert_eq!(s.merge_indices[i], added_index);
            assert_eq!(s.merge_radii[i], added_radius);
        }
    }

    #[test]
    fn test_add_too_small_merge() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.add_merge(1, 0.005f32);
        // The merge should not be added if the radius is too small
        assert_eq!(s.merge_indices[0], 0);
        assert_eq!(s.merge_radii[0], 0.0f32);
    }

    #[test]
    fn test_shift_1_merge() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        // This should technically not be possible by using public calls,
        // but for the sake of having the algorithm completely covered,
        // test the algorithm with some funky input
        s.merge_indices[MAX_MERGES - 1] = 1;
        s.merge_radii[MAX_MERGES - 1] = 1.0f32;
        s.shift_used_indices_left();
        assert_eq!(s.merge_indices[0], 1);
        assert_eq!(s.merge_radii[0], 1.0f32);
        for i in 1..MAX_MERGES {
            assert_eq!(s.merge_indices[i], 0);
            assert_eq!(s.merge_radii[i], 0.0f32);
        }
    }

    #[test]
    fn test_shift_2_merges() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        // This should technically not be possible by using public calls,
        // but for the sake of having the algorithm completely covered,
        // test the algorithm with some funky input
        s.merge_indices[MAX_MERGES - 1] = 1;
        s.merge_radii[MAX_MERGES - 1] = 1.0f32;
        s.merge_indices[MAX_MERGES - 2] = 2;
        s.merge_radii[MAX_MERGES - 2] = 2.0f32;
        s.shift_used_indices_left();
        assert_eq!(s.merge_indices[0], 2);
        assert_eq!(s.merge_radii[0], 2.0f32);
        assert_eq!(s.merge_indices[1], 1);
        assert_eq!(s.merge_radii[1], 1.0f32);
        for i in 2..MAX_MERGES {
            assert_eq!(s.merge_indices[i], 0);
            assert_eq!(s.merge_radii[i], 0.0f32);
        }
    }

    #[test]
    fn test_remove_merge() {
        // First same as adding merges
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        for i in 0..MAX_MERGES {
            let added_index = i as u32;
            let added_radius = (i + 1) as f32;
            s.add_merge(added_index, added_radius);
            assert_eq!(s.merge_indices[i], added_index);
            assert_eq!(s.merge_radii[i], added_radius);
        }
        // Now remove in the same order, which would be worst case scenario for the shifting algorithm
        for i in 0..MAX_MERGES {
            s.remove_merge(i as u32);
            for j in 0..MAX_MERGES {
                if j >= (MAX_MERGES - (i + 1)) {
                    assert_eq!(s.merge_indices[j], 0);
                    assert_eq!(s.merge_radii[j], 0.0f32);
                } else {
                    assert_eq!(s.merge_indices[j], (j + i + 1) as u32);
                    assert_eq!(s.merge_radii[j], (j + i + 2) as f32);
                }
            }
        }
    }

    #[test]
    fn test_set_radius() {
        let mut s = Primitive::new_sphere(DEFAULT_RADIUS);
        s.set_radius(50.0f32);
        // SAFETY: just created the primitive as a sphere, so sphere_data should be the correct payload
        unsafe {
            assert_eq!(s.payload.sphere_data.radius, 50.0f32);
        }
    }

    #[test]
    fn test_get_radius() {
        let s = Primitive::new_sphere(DEFAULT_RADIUS);
        let radius = s.get_radius();
        // SAFETY: just created the primitive as a sphere, so sphere_data should be the correct payload
        unsafe {
            assert_eq!(radius, s.payload.sphere_data.radius);
        }
    }
}