use ocl::OclPrm;

const MAX_OBJECTS_PER_NODE: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OctreeNode {
    pub subnode_indices: [u32; 8],
    pub indices: [u32; MAX_OBJECTS_PER_NODE],
}

// SAFETY: OctreeNode is plain-old-data (POD) and contains only OclPrm-compatible fields.
unsafe impl OclPrm for OctreeNode {}

impl OctreeNode {
    pub fn new() -> Self {
        Self {
            subnode_indices: [u32::MAX; 8],
            indices: [u32::MAX; MAX_OBJECTS_PER_NODE],
        }
    }

    pub fn add_item(&mut self, item_index: u32) { // TODO: Can't this be done more efficiently?
        // Find the first empty slot in indices
        if let Some(pos) = self.indices.iter().position(|&x| x == u32::MAX) {
            self.indices[pos] = item_index;
        }
    }

    pub fn remove_item(&mut self, item_index: u32) -> bool {
        // Find the item and set it to u32::MAX
        if let Some(pos) = self.indices.iter().position(|&x| x == item_index) {
            self.indices[pos] = u32::MAX;
            return true;
        }
        false
    }

    pub fn is_empty(&self) -> bool {
        self.indices.iter().all(|&x| x == u32::MAX) && self.subnode_indices.iter().all(|&x| x == u32::MAX)
    }
}

impl Default for OctreeNode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_default_octree_node() {
        let node = OctreeNode::default();
        assert!(node.is_empty());
        assert_eq!(node.subnode_indices, [u32::MAX; 8]);
        assert_eq!(node.indices, [u32::MAX; MAX_OBJECTS_PER_NODE]);
    }

    #[test]
    fn test_add_item_to_octree_node() {
        let mut node = OctreeNode::default();
        node.add_item(1);
        assert_eq!(node.indices[0], 1);
    }

    #[test]
    fn test_remove_item_from_octree_node() {
        let mut node = OctreeNode::default();
        node.add_item(1);
        assert!(node.remove_item(1));
        assert!(!node.remove_item(1));
    }

    #[test]
    fn test_is_empty_octree_node() {
        let node = OctreeNode::default();
        assert!(node.is_empty());
    }

    #[test]
    fn test_octree_node_not_empty_after_adding_item() {
        let mut node = OctreeNode::default();
        node.add_item(1);
        assert!(!node.is_empty());
    }
}

// #[derive(Default, Clone)]
// pub struct OctreeNode {
//     // Position of the node marks the bottom-left-front corner
//     pos_x: f32,
//     pos_y: f32,
//     pos_z: f32,
//     size: f32,
//     dirty: bool, // Indicates if the node needs to be updated
//     indices: Vec<u32>,
//     subnodes: [Option<Box<OctreeNode>>; 8],
//     serialized_index: usize, // Index in the serialized flat array
// }

// const SUBNODE_OFFSETS: [(f32, f32, f32); 8] = [
//     (0f32, 0f32, 0f32), // bottom-front-left
//     (1f32, 0f32, 0f32), // bottom-front-right
//     (0f32, 1f32, 0f32), // top-front-left
//     (1f32, 1f32, 0f32), // top-front-right
//     (0f32, 0f32, 1f32), // bottom-back-left
//     (1f32, 0f32, 1f32), // bottom-back-right
//     (0f32, 1f32, 1f32), // top-back-left
//     (1f32, 1f32, 1f32), // top-back-right
// ];

// const SIDE_COMP: [u8; 6] = [
//     0b10101010,
//     0b01010101,
//     0b11001100,
//     0b00110011,
//     0b11110000,
//     0b00001111,
// ];

// impl OctreeNode {
//     pub fn new(pos_x: f32, pos_y: f32, pos_z: f32, size: f32) -> Self {
//         Self {
//             pos_x,
//             pos_y,
//             pos_z,
//             size,
//             dirty: true, // Initially dirty
//             ..Default::default()
//         }
//     }

//     pub fn get_serialized_index(&self) -> usize {
//         self.serialized_index
//     }

//     pub fn move_object(&mut self,
//                        index: u32,
//                        object_pos_x: f32,
//                        object_pos_y: f32,
//                        object_pos_z: f32,
//                        object_size: f32,
//                        previous_pos: Option<(f32, f32, f32)>) {
//         // If the object size is smaller than half the current node size, we store it in subnode(s)
//         // Otherwise, we store it in the current node
//         // An object can be stored in multiple subnodes if it spans them

//         // Optionally handle previous position to move object out of subnode
//         let mut to_be_removed = true;
//         if let Some((_prev_x, _prev_y, _prev_z)) = previous_pos {
//             if let Some(pos) = self.indices.iter().position(|&x| x == index) {
//                 self.indices.swap_remove(pos);
//                 to_be_removed = false;
//             }
//         } else {
//             to_be_removed = false;
//         }

//         // If the object size is larger than half the current node size, it cannot fit in a subnode
//         let half_size = self.size / 2.0;
//         let center_x = self.pos_x + half_size;
//         let center_y = self.pos_y + half_size;
//         let center_z = self.pos_z + half_size;

//         let half_object_size = object_size / 2.0;
//         let obj_min_x = object_pos_x - half_object_size;
//         let obj_max_x = object_pos_x + half_object_size;
//         let obj_min_y = object_pos_y - half_object_size;
//         let obj_max_y = object_pos_y + half_object_size;
//         let obj_min_z = object_pos_z - half_object_size;
//         let obj_max_z = object_pos_z + half_object_size;
//         let outside_current_node = obj_min_x > self.pos_x + self.size ||
//             obj_max_x < self.pos_x ||
//             obj_min_y > self.pos_y + self.size ||
//             obj_max_y < self.pos_y ||
//             obj_min_z > self.pos_z + self.size ||
//             obj_max_z < self.pos_z;
//         if object_size > self.size / 2.0f32 && !outside_current_node {
//             self.indices.push(index);
//             self.dirty = true; // Mark the node as dirty since it has indices
//             return;
//         }

//         let (prev_obj_min_x, prev_obj_max_x, prev_obj_min_y, prev_obj_max_y, prev_obj_min_z, prev_obj_max_z) = if let Some((prev_x, prev_y, prev_z)) = previous_pos {
//             let half_object_size = object_size / 2.0;
//             (
//             prev_x - half_object_size,
//             prev_x + half_object_size,
//             prev_y - half_object_size,
//             prev_y + half_object_size,
//             prev_z - half_object_size,
//             prev_z + half_object_size,
//             )
//         } else {
//             (
//             obj_min_x,
//             obj_max_x,
//             obj_min_y,
//             obj_max_y,
//             obj_min_z,
//             obj_max_z,
//             )
//         };

//         // For each axis, determine which halves the object overlaps
//         let x_big = 0u8.wrapping_sub((obj_max_x > center_x) as u8);
//         let y_big = 0u8.wrapping_sub((obj_max_y > center_y) as u8);
//         let z_big = 0u8.wrapping_sub((obj_max_z > center_z) as u8);
//         let x_small = 0u8.wrapping_sub((obj_min_x < center_x) as u8);
//         let y_small = 0u8.wrapping_sub((obj_min_y < center_y) as u8);
//         let z_small = 0u8.wrapping_sub((obj_min_z < center_z) as u8);
//         let mask: u8 = 0b11111111 &
//             ((x_big & SIDE_COMP[0]) | (x_small & SIDE_COMP[1])) &
//             ((y_big & SIDE_COMP[2]) | (y_small & SIDE_COMP[3])) &
//             ((z_big & SIDE_COMP[4]) | (z_small & SIDE_COMP[5]));

//         let prev_x_big = 0u8.wrapping_sub((prev_obj_max_x > center_x) as u8);
//         let prev_y_big = 0u8.wrapping_sub((prev_obj_max_y > center_y) as u8);
//         let prev_z_big = 0u8.wrapping_sub((prev_obj_max_z > center_z) as u8);
//         let prev_x_small = 0u8.wrapping_sub((prev_obj_min_x < center_x) as u8);
//         let prev_y_small = 0u8.wrapping_sub((prev_obj_min_y < center_y) as u8);
//         let prev_z_small = 0u8.wrapping_sub((prev_obj_min_z < center_z) as u8);
//         let prev_mask: u8 = 0b11111111 &
//             ((prev_x_big & SIDE_COMP[0]) | (prev_x_small & SIDE_COMP[1])) &
//             ((prev_y_big & SIDE_COMP[2]) | (prev_y_small & SIDE_COMP[3])) &
//             ((prev_z_big & SIDE_COMP[4]) | (prev_z_small & SIDE_COMP[5]));
//         let prev_outside_current_node = prev_obj_min_x > self.pos_x + self.size ||
//             prev_obj_max_x < self.pos_x ||
//             prev_obj_min_y > self.pos_y + self.size ||
//             prev_obj_max_y < self.pos_y ||
//             prev_obj_min_z > self.pos_z + self.size ||
//             prev_obj_max_z < self.pos_z;

//         // Now, for each subnode, check if its position matches the mask
//         for i in 0..8 {
//             if to_be_removed && previous_pos.is_some() && !prev_outside_current_node {
//                 if (prev_mask >> i) & 1 != 0 {
//                     if let Some(subnode) = &mut self.subnodes[i] {
//                         subnode.move_object(index, object_pos_x, object_pos_y, object_pos_z, object_size, previous_pos);
//                         if subnode.is_empty() {
//                             self.subnodes[i] = None; // Remove empty subnode
//                             self.dirty = true; // Mark the node as dirty since a subnode was removed
//                         }
//                     }
//                 }
//             }

//             if (mask >> i) & 1 != 0 && !outside_current_node {
//                 if self.subnodes[i].is_none() {
//                     self.subnodes[i] = Some(Box::new(OctreeNode::new(
//                         self.pos_x + (SUBNODE_OFFSETS[i].0 * half_size),
//                         self.pos_y + (SUBNODE_OFFSETS[i].1 * half_size),
//                         self.pos_z + (SUBNODE_OFFSETS[i].2 * half_size),
//                         half_size,
//                     )));
//                     self.dirty = true; // Mark the node as dirty since a subnode was created
//                 }
//                 // Move the object into the appropriate subnode
//                 if let Some(subnode) = &mut self.subnodes[i] {
//                     if to_be_removed {
//                         subnode.move_object(index, object_pos_x, object_pos_y, object_pos_z, object_size, previous_pos);
//                     } else {
//                         // If nothing is to be removed, don't pass previous position so we don't unnecessarily check for removals
//                         subnode.move_object(index, object_pos_x, object_pos_y, object_pos_z, object_size, None);
//                     }
//                 }
//             }
//         }
//     }

//     pub fn is_empty(&self) -> bool {
//         // Check if the node is empty (no indices and no subnodes)
//         self.indices.is_empty() && self.subnodes.iter().all(|subnode| subnode.is_none())
//     }

//     pub fn serialize_node(&self) -> Vec<SerializedOctreeNode> {
//         let subnode_indices = self.subnodes.iter().map(|subnode| {
//             subnode.as_ref().map_or(u32::MAX, |s| s.serialized_index as u32)
//         }).collect::<Vec<_>>();
//         SerializedOctreeNode {
//             self.subnodes
//         }
//     }

//     /// Serializes the octree node into a flat array suitable for GPU consumption.
//     /// Each node is serialized as:
//     /// [pos_x, pos_y, pos_z, size, indices_start, indices_count, subnode0, subnode1, ..., subnode7]
//     /// where subnodeN is the index in the flat array of the child node, or u32::MAX if None.
//     /// The indices themselves are stored in a separate flat array.
//     pub fn serialize_flat(
//         &self,
//         nodes: &mut Vec<[f32; 14]>,
//         indices: &mut Vec<u32>,
//     ) -> usize {
//         let node_index = nodes.len();

//         // Reserve space for this node
//         nodes.push([0.0; 14]);

//         // Store indices for this node
//         let indices_start = indices.len() as u32;
//         let indices_count = self.indices.len() as u32;
//         indices.extend_from_slice(&self.indices);

//         // Prepare subnode indices (u32::MAX means None)
//         let mut subnode_indices = [u32::MAX; 8];
//         for (i, subnode) in self.subnodes.iter().enumerate() {
//             if let Some(child) = subnode {
//                 let child_index = child.serialize_flat(nodes, indices);
//                 subnode_indices[i] = child_index as u32;
//             }
//         }

//         // Write node data
//         nodes[node_index] = [
//             self.pos_x,
//             self.pos_y,
//             self.pos_z,
//             self.size,
//             indices_start as f32,
//             indices_count as f32,
//             subnode_indices[0] as f32,
//             subnode_indices[1] as f32,
//             subnode_indices[2] as f32,
//             subnode_indices[3] as f32,
//             subnode_indices[4] as f32,
//             subnode_indices[5] as f32,
//             subnode_indices[6] as f32,
//             subnode_indices[7] as f32,
//         ];

//         node_index
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_new_node_defaults() {
//         let node = OctreeNode::new(1.0, 2.0, 3.0, 4.0);
//         assert_eq!(node.pos_x, 1.0);
//         assert_eq!(node.pos_y, 2.0);
//         assert_eq!(node.pos_z, 3.0);
//         assert_eq!(node.size, 4.0);
//         assert!(node.indices.is_empty());
//         assert!(node.subnodes.iter().all(|n| n.is_none()));
//     }

//     #[test]
//     fn test_move_object_stored_in_node() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         // Object too large for subnodes, should be stored in root node
//         node.move_object(42, 0.5, 0.5, 0.5, 1.5, None);
//         assert_eq!(node.indices, vec![42]);
//     }

//     #[test]
//     fn test_move_object_stored_in_node_to_other_position_inside_node() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         // Object too large for subnodes, should be stored in root node
//         node.move_object(42, 0.5, 0.5, 0.5, 1.5, None);
//         assert_eq!(node.indices, vec![42]);
//         // Move the object to a new position inside the same node, the current node should still hold it
//         node.move_object(42, 0.6, 0.6, 0.6, 1.5, Some((0.5, 0.5, 0.5)));
//         assert_eq!(node.indices, vec![42]);
//     }

//     #[test]
//     fn test_move_object_stored_in_node_to_other_node() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         // Object too large for subnodes, should be stored in root node
//         node.move_object(42, 0.5, 0.5, 0.5, 1.5, None);
//         assert_eq!(node.indices, vec![42]);
//         // Now set the position outside of the current node and check if it gets removed
//         node.move_object(42, 3.0, 3.0, 3.0, 1.5, Some((0.5, 0.5, 0.5)));
//         assert!(node.indices.is_empty());
//     }

//     #[test]
//     fn test_move_object_stored_in_subnode() {
//         for (i, (dx, dy, dz)) in SUBNODE_OFFSETS.iter().enumerate() {
//             let node_size: f32 = 2.0;
//             let mut node = OctreeNode::new(0.0, 0.0, 0.0, node_size);
//             // Object small enough for subnodes
//             node.move_object(7, dx * node_size + 0.1f32, dy * node_size + 0.1f32, dz * node_size + 0.1f32, 0.6, None);
//             // Should not be in root node
//             assert!(node.indices.is_empty());
//             // Should be in the correct subnode
//             for j in 0..8 {
//                 if i == j {
//                     assert!(node.subnodes[j].is_some());
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, dx * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, dy * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, dz * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//                 } else {
//                     assert!(node.subnodes[j].is_none());
//                 }
//             }
//         }
//     }

//     #[test]
//     fn test_move_object_stored_in_subnode_to_other_subnode() {
//         let node_size: f32 = 2.0;
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, node_size);
//         // Object small enough for subnodes
//         node.move_object(7, 0.1, 0.1, 0.1, 0.6, None);
//         // Should not be in root node
//         assert!(node.indices.is_empty());
//         // Should be in the correct subnode
//         for j in 0..8 {
//             if 0 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else {
//                 assert!(node.subnodes[j].is_none());
//             }
//         }
//         // Now move the object to a different subnode
//         node.move_object(7, 1.5, 1.5, 1.5, 0.6, Some((0.1, 0.1, 0.1)));
//         // Should not be in root node
//         assert!(node.indices.is_empty());
//         // Should be in the correct subnode
//         for j in 0..8 {
//             if 7 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else {
//                 assert!(node.subnodes[j].is_none());
//             }
//         }
//     }

//     #[test]
//     fn test_move_object_stored_in_multiple_subnodes() {
//         // Take the first 4 offsets, which should all be the back subnodes
//         for (i, (dx, dy, dz)) in SUBNODE_OFFSETS.iter().take(4).enumerate() {
//             let node_size: f32 = 2.0;
//             let mut node = OctreeNode::new(0.0, 0.0, 0.0, node_size);
//             // Object small enough for subnodes
//             node.move_object(7, dx * node_size + 0.1f32, dy * node_size + 0.1f32, node_size / 2.0, 0.6, None);
//             // Should not be in root node
//             assert!(node.indices.is_empty());
//             // Should be in the correct subnodes
//             for j in 0..8 {
//                 if i == j {
//                     assert!(node.subnodes[j].is_some());
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, dx * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, dy * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, dz * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//                 } else if i + 4 == j {
//                     assert!(node.subnodes[j].is_some());
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, dx * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, dy * node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, dz * node_size / 2.0 + node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                     assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//                 } else {
//                     assert!(node.subnodes[j].is_none());
//                 }
//             }
//         }
//     }

//     #[test]
//     fn test_move_object_stored_in_multiple_subnodes_to_other_multiple_subnodes() {
//         let node_size: f32 = 2.0;
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, node_size);
//         // Object small enough for subnodes
//         node.move_object(7, 0.1f32, 0.1f32, node_size / 2.0, 0.6, None);
//         // Should not be in root node
//         assert!(node.indices.is_empty());
//         // Should be in the correct subnodes
//         for j in 0..8 {
//             if 0 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else if 4 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, 0.0 + node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else {
//                 assert!(node.subnodes[j].is_none());
//             }
//         }
//         // Now move the object to a different set of subnodes
//         node.move_object(7, 0.1f32, node_size * 0.75f32, node_size / 2.0, 0.6, Some((0.1, 0.1, node_size / 2.0)));
//         // Should not be in root node
//         assert!(node.indices.is_empty());
//         for j in 0..8 {
//             if 2 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else if 6 == j {
//                 assert!(node.subnodes[j].is_some());
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_x, 0.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_y, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().pos_z, 0.0 + node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().size, node_size / 2.0);
//                 assert_eq!(node.subnodes[j].as_ref().unwrap().indices, vec![7]);
//             } else {
//                 assert!(node.subnodes[j].is_none());
//             }
//         }
//     }

//     #[test]
//     fn test_move_object_out_of_empty_node() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         // Move an object that is not in the node
//         node.move_object(42, 3.0, 3.0, 3.0, 1.5, None);
//         // The node should still be empty
//         assert!(node.is_empty());
//     }

//     #[test]
//     fn test_move_object_multiple_levels_down() {
//         // A test that incrementally makes a smaller object that should go down multiple levels as it grows smaller
//         const MAX_LEVELS: usize = 10;
//         let base_size = 256.0;
//         // Move an object that should go down multiple levels
//         for level in 1..MAX_LEVELS {
//             let object_size = (base_size - 0.1) / (2.0_f32.powi(level as i32));
//             let mut base_node = OctreeNode::new(0.0, 0.0, 0.0, base_size);
//             base_node.move_object(42, 0.0, 0.0, 0.0, object_size, None);
//             let mut current_node = &base_node;
//             for l in 0..level {
//                 for n in 0..8 {
//                     if n == 0 {
//                         // The first subnode should always contain the object
//                         if let Some(subnode) = &current_node.subnodes[n] {
//                             assert_eq!(subnode.size, base_size / (2.0_f32.powi(l as i32 + 1)));
//                             assert_eq!(subnode.pos_x, 0.0);
//                             assert_eq!(subnode.pos_y, 0.0);
//                             assert_eq!(subnode.pos_z, 0.0);
//                             if l == level - 1 {
//                                 // The last level should contain the object
//                                 assert_eq!(subnode.indices, vec![42]);
//                             } else {
//                                 // Other levels should not contain the object
//                                 assert!(subnode.indices.is_empty());
//                             }
//                         }
//                     } else {
//                         // Other subnodes should be empty
//                         assert!(current_node.subnodes[n].is_none());
//                     }
//                 }
//                 current_node = current_node.subnodes[0].as_ref().unwrap();
//             }
//         }
//     }

//     #[test]
//     fn test_is_empty_true_for_new_node() {
//         let node = OctreeNode::new(0.0, 0.0, 0.0, 1.0);
//         assert!(node.is_empty());
//     }

//     #[test]
//     fn test_is_empty_false_when_object_added() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         node.move_object(5, 1.0, 1.0, 1.0, 2.0, None);
//         assert!(!node.is_empty());
//     }

//     #[test]
//     fn test_serialize_flat_basic() {
//         let mut node = OctreeNode::new(0.0, 0.0, 0.0, 2.0);
//         node.move_object(10, 1.0, 1.0, 1.0, 2.0, None);
//         let mut nodes = Vec::new();
//         let mut indices = Vec::new();
//         let root_index = node.serialize_flat(&mut nodes, &mut indices);
//         assert_eq!(root_index, 0);
//         assert_eq!(indices, vec![10]);
//         assert_eq!(nodes.len(), 1);
//         assert_eq!(nodes[0][0], 0.0); // pos_x
//         assert_eq!(nodes[0][3], 2.0); // size
//         assert_eq!(nodes[0][4], 0.0); // indices_start
//         assert_eq!(nodes[0][5], 1.0); // indices_count
//     }
// }