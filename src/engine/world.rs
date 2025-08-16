use crate::engine::sphere_merge::MergeModel;
use crate::engine::lights::directionlight::DirectionLight;
use crate::engine::util::octree::octree::Octree;
use crate::engine::primitives::primitive::Primitive;

const MAX_MERGES: u32 = 6;

pub struct World {
    primitives: Vec<Primitive>,
    merge_models: Vec<MergeModel>,
    directionlight: DirectionLight,
    octree: Octree,
}

impl World {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            merge_models: Vec::new(),
            directionlight: DirectionLight::default(),
            octree: Octree::new(64f32, 3, 3, 3, (-64f32, -64f32, -64f32)),
         }
    }

    pub fn get_merges(&mut self) -> (Vec<u32>, Vec<f32>) {
        let mut merge_indices = Vec::<u32>::new();
        let mut merge_radii = Vec::<f32>::new();

        for merge_model in self.merge_models.iter_mut() {
            let current_count = self.primitives.len() as u32;
            for i in 0..merge_model.get_primitives().len() as u32 {
                let mut added_merges = 0u32;
                for merge in merge_model.get_merges().iter() {
                    if added_merges < MAX_MERGES {
                        if merge.get_sphere1() == i as u32 {
                            merge_indices.push((merge.get_sphere2()) as u32);
                            merge_radii.push(merge.get_inverted_radius());
                            added_merges += 1;
                        } else if merge.get_sphere2() == i as u32 {
                            merge_indices.push((merge.get_sphere1()) as u32);
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
        (merge_indices, merge_radii)
    }

    pub fn push_primitive(&mut self, primitive: Primitive) {
        self.primitives.push(primitive);
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

    pub fn get_primitives(&self) -> &Vec<Primitive> {
        &self.primitives
    }
}