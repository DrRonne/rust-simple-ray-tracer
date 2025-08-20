use crate::engine::lights::directionlight::DirectionLight;
use crate::engine::util::octree::octree::Octree;
use crate::engine::util::tombstoned_list::TombstonedList;
use crate::engine::primitives::primitive::Primitive;

const MAX_MERGES: u32 = 6;

pub struct World {
    primitives: TombstonedList<Primitive>,
    directionlight: DirectionLight,
    octree: Octree,
}

impl World {
    pub fn new() -> Self {
        Self {
            primitives: TombstonedList::new(Primitive::default()),
            directionlight: DirectionLight::default(),
            octree: Octree::new(64f32, 3, 3, 3, (-64f32, -64f32, -64f32)),
         }
    }

    pub fn push_primitive(&mut self, primitive: Primitive) -> usize {
        self.primitives.add(primitive)
    }

    pub fn get_direction_light_direction_vec(&self) ->[f32; 3] {
        self.directionlight.get_direction()
    }

    pub fn get_direction_light_color_vec(&self) -> [u8; 3] {
        self.directionlight.get_color()
    }

    pub fn set_direction_light_direction(&mut self, direction: [f32; 3]) {
        self.directionlight.set_direction(direction);
    }

    pub fn get_primitives(&self) -> &Vec<Primitive> {
        self.primitives.get_items()
    }

    pub fn merge_primitives(&mut self, index1: usize, index2: usize, radius: f32) {
        // This is written in this way because you can't have 2 mutable references to self.primitives
        // I'm sure there is a better way somehow, but for now we just remove the index from the first object again if it turns out the second one doesn't exist.
        if let Some(p1) = self.primitives.get_mut(index1) {
            p1.add_merge(index2 as u32, radius);
        } else {
            println!("WARNING: Attempting to merge primitive (index 1) that does not exist!");
        }
        if let Some(p2) = self.primitives.get_mut(index2) {
            p2.add_merge(index1 as u32, radius);
        } else {
            println!("WARNING: Attempting to merge primitive (index 2) that does not exist!");
            if let Some(p1) = self.primitives.get_mut(index1) {
                // Should remove it again from the first one
                p1.remove_merge(index2 as u32);
            }
        }
    }
}