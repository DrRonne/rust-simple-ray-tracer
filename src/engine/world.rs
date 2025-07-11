use crate::engine::render::{Renderable, RenderObject};
use crate::engine::sphere_merge::{MergeModel, SphereMerge};
use crate::engine::lights::directionlight::DirectionLight;

const MAX_MERGES: u32 = 6;

#[derive(Default)]
pub struct World {
    objects: Vec<Box<dyn Renderable>>,
    merge_models: Vec<MergeModel>,
    directionlight: DirectionLight,
}

impl World {
    pub fn new() -> Self {
        Self {
            ..Default::default()
         }
    }

    pub fn get_render_objects_and_merges(&mut self) -> (Vec<RenderObject>, Vec<u32>, Vec<f32>) {
        let mut render_objects: Vec<RenderObject> = Vec::new();
        let mut merge_indices = Vec::<u32>::new();
        let mut merge_radii = Vec::<f32>::new();
        for obj in self.objects.iter_mut() {
            render_objects.push(obj.get_render_object());
            for _ in 0..MAX_MERGES {
                merge_indices.push(render_objects.len() as u32 - 1 as u32);
                merge_radii.push(0.0f32);
            }
        }
        for merge_model in self.merge_models.iter_mut() {
            let current_count = render_objects.len() as u32;
            render_objects.append(&mut merge_model.get_render_objects());
            for i in 0..merge_model.get_render_objects().len() as u32 {
                let mut added_merges = 0u32;
                for merge in merge_model.get_merges().iter() {
                    if added_merges < MAX_MERGES {
                        if merge.get_sphere1() == i as u32 {
                            merge_indices.push((current_count + merge.get_sphere2()) as u32);
                            merge_radii.push(merge.get_inverted_radius());
                            added_merges += 1;
                        } else if merge.get_sphere2() == i as u32 {
                            merge_indices.push((current_count + merge.get_sphere1()) as u32);
                            merge_radii.push(merge.get_inverted_radius());
                            added_merges += 1;
                        }
                    }
                }
                for _ in added_merges..MAX_MERGES {
                    merge_indices.push((current_count + i) as u32);
                    merge_radii.push(0.0f32);
                }
            }
        }
        return (render_objects, merge_indices, merge_radii);
    }

    pub fn push_renderable(&mut self, render_object: Box<dyn Renderable>) {
        self.objects.push(render_object);
    }

    pub fn push_merge_model(&mut self, merge_model: MergeModel) {
        self.merge_models.push(merge_model);
    }

    pub fn get_direction_light_direction_vec(&mut self) -> Vec<f32> {
        return self.directionlight.get_direction();
    }

    pub fn get_direction_light_color_vec(&mut self) -> Vec<u8> {
        return self.directionlight.get_color();
    }
}