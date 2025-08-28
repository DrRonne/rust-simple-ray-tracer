use crate::engine::lights::directionlight::DirectionLight;
use crate::engine::util::octree::{octree::Octree, octree_node::OctreeNode};
use crate::engine::util::tombstoned_list::TombstonedList;
use crate::engine::primitives::primitive::Primitive;
use crate::engine::util::cframe::Positionable;

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
        let index = self.primitives.add(primitive);
        self.octree.move_item(index as u32, primitive.get_position().0, primitive.get_position().1, primitive.get_position().2, primitive.get_render_radius(), None);
        index
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

    pub fn set_primitive_position(&mut self, index: usize, position: [f32; 3]) {
        if let Some(primitive) = self.primitives.get_mut(index) {
            self.octree.move_item(index as u32, position[0], position[1], position[2], primitive.get_render_radius(), Some(primitive.get_position()));
            primitive.set_position(position[0], position[1], position[2]);
        } else {
            println!("WARNING: Attempting to set position of primitive (index {}) that does not exist!", index);
        }
    }

    pub fn get_octree_nodes(&self) -> &Vec<OctreeNode> {
        self.octree.get_nodes().get_items()
    }

    pub fn get_octree_root_position(&self) -> (f32, f32, f32) {
        self.octree.get_root_position()
    }

    pub fn get_octree_root_node_indices(&self) -> &Vec<u32> {
        self.octree.get_root_node_indices()
    }

    pub fn get_octree_root_node_size(&self) -> f32 {
        self.octree.get_root_node_size()
    }

    pub fn get_octree_dimensions(&self) -> (u32, u32, u32) {
        (self.octree.get_width(), self.octree.get_height(), self.octree.get_depth())
    }
}