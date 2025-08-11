use crate::engine::render::render::{Renderable, RenderObject};

#[derive(Default)]
pub struct MergeModel {
    spheres: Vec<Box<dyn Renderable>>,
    merges: Vec<SphereMerge>
}

#[derive(Default)]
pub struct SphereMerge {
    sphere1: u32,
    sphere2: u32,
    inverted_radius: f32,
}

impl MergeModel {
    pub fn new() -> Self {
        Self {
            ..Default::default()
         }
    }

    pub fn add_sphere(&mut self, sphere: Box<dyn Renderable>) {
        self.spheres.push(sphere);
    }

    pub fn add_merge(&mut self, sphere1: u32, sphere2: u32, inverted_radius: f32) {
        self.merges.push(
            SphereMerge {
                sphere1,
                sphere2,
                inverted_radius,
            }
        );
    }

    pub fn get_render_objects(&mut self) -> Vec<RenderObject> {
        let mut render_objects: Vec<RenderObject> = Vec::new();
        for obj in self.spheres.iter_mut() {
            render_objects.push(obj.get_render_object());
        }
        return render_objects;
    }

    pub fn get_merges(&self) -> &Vec<SphereMerge> {
        &self.merges
    }

    pub fn copy_merges_with_offset(&self, offset: u32) -> Vec<SphereMerge> {
        let mut copy_merges: Vec<SphereMerge> = Vec::new();
        for merge in self.merges.iter() {
            copy_merges.push(SphereMerge {
                sphere1: merge.sphere1 + offset,
                sphere2: merge.sphere2 + offset,
                inverted_radius: merge.inverted_radius,
            });
        }
        return copy_merges;
    }
}

impl SphereMerge {
    pub fn new(sphere1: u32, sphere2: u32, inverted_radius: f32) -> Self {
        Self {
            sphere1,
            sphere2,
            inverted_radius,
        }
    }

    pub fn get_sphere1(&self) -> u32 {
        self.sphere1
    }

    pub fn get_sphere2(&self) -> u32 {
        self.sphere2
    }

    pub fn get_inverted_radius(&self) -> f32 {
        self.inverted_radius
    }

    pub fn set_inverted_radius(&mut self, inverted_radius: f32) {
        self.inverted_radius = inverted_radius;
    }

    pub fn set_sphere1(&mut self, sphere1: u32) {
        self.sphere1 = sphere1;
    }

    pub fn set_sphere2(&mut self, sphere2: u32) {
        self.sphere2 = sphere2;
    }
}
