use std::collections::HashMap;

use crate::engine::util::octree::octree_node::OctreeNode;
use crate::engine::util::tombstoned_list::TombstonedList;

// Technically, this is not an octree but a collection of octree nodes.
// It can contain multiple root nodes, each representing a separate octree structure.
// This allows for more flexibility in managing multiple octrees within a single structure.
pub struct Octree {
    root_node_size: f32,
    width: u32, // How many chunks of root nodes are in the x direction
    height: u32, // How many chunks of root nodes are in the y direction
    depth: u32, // How many chunks of root nodes are in the z direction
    nodes: TombstonedList<OctreeNode>,
    root_node_indices: Vec<u32>, // u32 for GPU compatibility
    root_position: (f32, f32, f32), // Bottom left back position of the octree (chunks) in world space
}

const SUBNODE_OFFSETS: [(f32, f32, f32); 8] = [
    (0f32, 0f32, 0f32), // bottom-front-left
    (1f32, 0f32, 0f32), // bottom-front-right
    (0f32, 1f32, 0f32), // top-front-left
    (1f32, 1f32, 0f32), // top-front-right
    (0f32, 0f32, 1f32), // bottom-back-left
    (1f32, 0f32, 1f32), // bottom-back-right
    (0f32, 1f32, 1f32), // top-back-left
    (1f32, 1f32, 1f32), // top-back-right
];

const SIDE_COMP: [u8; 6] = [
    0b10101010,
    0b01010101,
    0b11001100,
    0b00110011,
    0b11110000,
    0b00001111,
];

impl Octree {
    pub fn new(
        root_node_size: f32,
        width: u32,
        height: u32,
        depth: u32,
        root_position: (f32, f32, f32)
    ) -> Self {
        let total_chunks = width * height * depth;
        let mut root_node_indices = vec![u32::MAX; total_chunks as usize];
        let mut nodes = TombstonedList::new(OctreeNode::default());
        for i in 0..total_chunks {
            if i >= u32::MAX as u32 {
                panic!("Too many chunks for a single octree, maximum is {}", u32::MAX);
            }
            let node = OctreeNode::new();
            let index = nodes.add(node);
            root_node_indices[i as usize] = index as u32;
        }
        
        Octree {
            root_node_size,
            width,
            height,
            depth,
            nodes,
            root_node_indices,
            root_position,
        }
    }

    pub fn get_nodes(&self) -> &TombstonedList<OctreeNode> {
        &self.nodes
    }

    pub fn get_root_node_indices(&self) -> &Vec<u32> {
        &self.root_node_indices
    }

    pub fn get_root_position(&self) -> (f32, f32, f32) {
        self.root_position
    }

    pub fn get_root_node_size(&self) -> f32 {
        self.root_node_size
    }

    pub fn get_width(&self) -> u32 {
        self.width
    }

    pub fn get_height(&self) -> u32 {
        self.height
    }

    pub fn get_depth(&self) -> u32 {
        self.depth
    }

    pub fn get_root_node_by_position(
        &self,
        pos_x: f32,
        pos_y: f32,
        pos_z: f32
    ) -> Option<(usize, f32, f32, f32)> {
        let (root_pos_x, root_pos_y, root_pos_z) = self.root_position;
        let node_size = self.root_node_size;
        
        // Calculate the index based on the position
        let index_x = ((pos_x - root_pos_x) / node_size).floor() as u32;
        let index_y = ((pos_y - root_pos_y) / node_size).floor() as u32;
        let index_z = ((pos_z - root_pos_z) / node_size).floor() as u32;

        if index_x < self.width && index_y < self.height && index_z < self.depth {
            Some(((index_x + index_y * self.width + index_z * self.width * self.height) as usize, root_pos_x + self.root_node_size * index_x as f32, root_pos_y + self.root_node_size * index_y as f32, root_pos_z + self.root_node_size * index_z as f32))
        } else {
            None
        }
    }

    pub fn get_root_nodes_by_position_and_size(
        &self,
        pos_x: f32,
        pos_y: f32,
        pos_z: f32,
        size: f32,
    ) -> HashMap<u32, (f32, f32, f32)> {
        let (root_pos_x, root_pos_y, root_pos_z) = self.root_position;
        let node_size = self.root_node_size;

        // Calculate the index based on the position
        let lower_x = ((pos_x - (size / 2.0) - root_pos_x) / node_size).floor() as u32;
        let lower_y = ((pos_y - (size / 2.0) - root_pos_y) / node_size).floor() as u32;
        let lower_z = ((pos_z - (size / 2.0) - root_pos_z) / node_size).floor() as u32;
        let upper_x = ((pos_x + (size / 2.0) - root_pos_x) / node_size).floor() as u32;
        let upper_y = ((pos_y + (size / 2.0) - root_pos_y) / node_size).floor() as u32;
        let upper_z = ((pos_z + (size / 2.0) - root_pos_z) / node_size).floor() as u32;
        // Gather all root nodes that overlap with the given position and size
        let mut results = HashMap::new();

        for z in lower_z..=upper_z {
            for y in lower_y..=upper_y {
                for x in lower_x..=upper_x {
                    if x >= self.width || y >= self.height || z >= self.depth {
                        continue; // Skip if the index is out of bounds
                    }
                    let flat_index = x + y * self.width + z * self.width * self.height;
                    results.insert(flat_index, (root_pos_x + self.root_node_size * x as f32, root_pos_y + self.root_node_size * y as f32, root_pos_z + self.root_node_size * z as f32));
                    // results.push((flat_index as usize, root_pos_x + self.root_node_size * x as f32, root_pos_y + self.root_node_size * y as f32, root_pos_z + self.root_node_size * z as f32));
                }
            }
        }

        results
    }

    fn move_item_in_node(
        &mut self,
        item_id: u32,
        object_pos_x: f32,
        object_pos_y: f32,
        object_pos_z: f32,
        object_size: f32,
        previous_pos: Option<(f32, f32, f32)>,
        node_index: usize,
        node_pos_x: f32,
        node_pos_y: f32,
        node_pos_z: f32,
        node_size: f32,
        to_be_added: bool
    ) {
        // Optionally handle previous position to move object out of subnode
        let mut to_be_removed = true;
        let mut to_be_added_mut = to_be_added;
        let node = self.nodes.get_mut(node_index).expect("Node should exist");
        if let Some((_prev_x, _prev_y, _prev_z)) = previous_pos {
            if node.remove_item(item_id) {
                to_be_removed = false;
            }
        } else {
            to_be_removed = false;
        }

        // let node = self.nodes.get_mut(node_index).expect("Node should exist");
        let half_size = node_size / 2.0;
        let center_x = node_pos_x + half_size;
        let center_y = node_pos_y + half_size;
        let center_z = node_pos_z + half_size;

        let half_object_size = object_size / 2.0;
        let obj_min_x = object_pos_x - half_object_size;
        let obj_max_x = object_pos_x + half_object_size;
        let obj_min_y = object_pos_y - half_object_size;
        let obj_max_y = object_pos_y + half_object_size;
        let obj_min_z = object_pos_z - half_object_size;
        let obj_max_z = object_pos_z + half_object_size;
        let outside_current_node = obj_min_x > node_pos_x + node_size ||
            obj_max_x < node_pos_x ||
            obj_min_y > node_pos_y + node_size ||
            obj_max_y < node_pos_y ||
            obj_min_z > node_pos_z + node_size ||
            obj_max_z < node_pos_z;
        if to_be_added && object_size > node_size / 2.0f32 && !outside_current_node {
            node.add_item(item_id);
            to_be_added_mut = false;
            if !to_be_removed {
                // If the item is not being removed, we can return early
                return;
            }
        }

        let (prev_obj_min_x, prev_obj_max_x, prev_obj_min_y, prev_obj_max_y, prev_obj_min_z, prev_obj_max_z) = if let Some((prev_x, prev_y, prev_z)) = previous_pos {
            let half_object_size = object_size / 2.0;
            (
            prev_x - half_object_size,
            prev_x + half_object_size,
            prev_y - half_object_size,
            prev_y + half_object_size,
            prev_z - half_object_size,
            prev_z + half_object_size,
            )
        } else {
            (
            obj_min_x,
            obj_max_x,
            obj_min_y,
            obj_max_y,
            obj_min_z,
            obj_max_z,
            )
        };

        let prev_x_big = 0u8.wrapping_sub((prev_obj_max_x > center_x) as u8);
        let prev_y_big = 0u8.wrapping_sub((prev_obj_max_y > center_y) as u8);
        let prev_z_big = 0u8.wrapping_sub((prev_obj_max_z > center_z) as u8);
        let prev_x_small = 0u8.wrapping_sub((prev_obj_min_x < center_x) as u8);
        let prev_y_small = 0u8.wrapping_sub((prev_obj_min_y < center_y) as u8);
        let prev_z_small = 0u8.wrapping_sub((prev_obj_min_z < center_z) as u8);
        let prev_mask: u8 = 0b11111111 &
            ((prev_x_big & SIDE_COMP[0]) | (prev_x_small & SIDE_COMP[1])) &
            ((prev_y_big & SIDE_COMP[2]) | (prev_y_small & SIDE_COMP[3])) &
            ((prev_z_big & SIDE_COMP[4]) | (prev_z_small & SIDE_COMP[5]));
        let prev_outside_current_node = prev_obj_min_x > node_pos_x + node_size ||
            prev_obj_max_x < node_pos_x ||
            prev_obj_min_y > node_pos_y + node_size ||
            prev_obj_max_y < node_pos_y ||
            prev_obj_min_z > node_pos_z + node_size ||
            prev_obj_max_z < node_pos_z;

        // For each axis, determine which halves the object overlaps
        let x_big = 0u8.wrapping_sub((obj_max_x > center_x) as u8);
        let y_big = 0u8.wrapping_sub((obj_max_y > center_y) as u8);
        let z_big = 0u8.wrapping_sub((obj_max_z > center_z) as u8);
        let x_small = 0u8.wrapping_sub((obj_min_x < center_x) as u8);
        let y_small = 0u8.wrapping_sub((obj_min_y < center_y) as u8);
        let z_small = 0u8.wrapping_sub((obj_min_z < center_z) as u8);
        let mask: u8 = 0b11111111 &
            ((x_big & SIDE_COMP[0]) | (x_small & SIDE_COMP[1])) &
            ((y_big & SIDE_COMP[2]) | (y_small & SIDE_COMP[3])) &
            ((z_big & SIDE_COMP[4]) | (z_small & SIDE_COMP[5]));

        // Now, for each subnode, check if its position matches the mask
        for i in 0..8 {
            // First check to remove the object from the previous position if it exists
            if to_be_removed && previous_pos.is_some() && !prev_outside_current_node {
                if (prev_mask >> i) & 1 != 0 {
                    let subnode_index = self.nodes.get(node_index).expect("Node should exist").subnode_indices[i];
                    if subnode_index != u32::MAX {
                        self.move_item_in_node(
                            item_id,
                            object_pos_x,
                            object_pos_y,
                            object_pos_z,
                            object_size,
                            previous_pos,
                            subnode_index as usize,
                            node_pos_x + (i & 1) as f32 * half_size,
                            node_pos_y + ((i >> 1) & 1) as f32 * half_size,
                            node_pos_z + ((i >> 2) & 1) as f32 * half_size,
                            half_size,
                            to_be_added_mut,
                        );
                        if self.nodes.get(subnode_index as usize).expect("Node should exist").is_empty() {
                            self.nodes.remove(subnode_index as usize);
                            self.nodes.get_mut(node_index).expect("Node should exist").subnode_indices[i] = u32::MAX;
                        }
                    }
                }
            }
            if (mask >> i) & 1 != 0 && !outside_current_node {
                if !to_be_added_mut {
                    // If nothing is to be added, we can skip the rest of the logic
                    continue;
                }

                let needs_new_subnode = self.nodes.get(node_index).expect("Node should exist").subnode_indices[i] == u32::MAX;
                if needs_new_subnode {
                    let added_index = self.nodes.add(OctreeNode::new()) as u32;
                    self.nodes.get_mut(node_index).expect("Node should exist").subnode_indices[i] = added_index;
                }
                // Move the object into the appropriate subnode
                if to_be_removed {
                    self.move_item_in_node(
                        item_id,
                        object_pos_x,
                        object_pos_y,
                        object_pos_z,
                        object_size,
                        previous_pos,
                        self.nodes.get(node_index).expect("Node should exist").subnode_indices[i] as usize,
                        node_pos_x + (i & 1) as f32 * half_size,
                        node_pos_y + ((i >> 1) & 1) as f32 * half_size,
                        node_pos_z + ((i >> 2) & 1) as f32 * half_size,
                        half_size,
                        to_be_added_mut,
                    );
                } else {
                    // If nothing is to be removed, don't pass previous position so we don't unnecessarily check for removals
                    self.move_item_in_node(
                        item_id,
                        object_pos_x,
                        object_pos_y,
                        object_pos_z,
                        object_size,
                        None,
                        self.nodes.get(node_index).expect("Node should exist").subnode_indices[i] as usize,
                        node_pos_x + (i & 1) as f32 * half_size,
                        node_pos_y + ((i >> 1) & 1) as f32 * half_size,
                        node_pos_z + ((i >> 2) & 1) as f32 * half_size,
                        half_size,
                        to_be_added_mut,
                    );
                }
            }
        }
    }

    pub fn move_item(
        &mut self,
        item_id: u32,
        object_pos_x: f32,
        object_pos_y: f32,
        object_pos_z: f32,
        object_size: f32,
        previous_pos: Option<(f32, f32, f32)>
    ) {
        let remove_root_nodes = previous_pos.map_or(HashMap::new(), |(prev_x, prev_y, prev_z)| {
            self.get_root_nodes_by_position_and_size(prev_x, prev_y, prev_z, object_size)
        });
        let mut add_root_nodes = self.get_root_nodes_by_position_and_size(object_pos_x, object_pos_y, object_pos_z, object_size);
        let node_size = self.root_node_size;
        for (root_index, (node_pos_x, node_pos_y, node_pos_z)) in remove_root_nodes {
            // Remove the item from the previous root node
            // if the root index is also in the to be added root nodes, set to_be_added to true
            let to_be_added = add_root_nodes.remove(&root_index).is_some();
            if to_be_added {
                self.move_item_in_node(
                    item_id,
                    object_pos_x,
                    object_pos_y,
                    object_pos_z,
                    object_size,
                    previous_pos,
                    root_index as usize,
                    node_pos_x,
                    node_pos_y,
                    node_pos_z,
                    node_size,
                    true,
                );
            } else {
                self.move_item_in_node(
                    item_id,
                    f32::MAX,
                    f32::MAX,
                    f32::MAX,
                    object_size,
                    previous_pos,
                    root_index as usize,
                    node_pos_x,
                    node_pos_y,
                    node_pos_z,
                    node_size,
                    false,
                );
            }
        }
        for (root_index, (node_pos_x, node_pos_y, node_pos_z)) in add_root_nodes {
            // Remove the item from the previous root node
            // if the root index is also in the to be added root nodes, set to_be_added to true
            self.move_item_in_node(
                item_id,
                object_pos_x,
                object_pos_y,
                object_pos_z,
                object_size,
                None,
                root_index as usize,
                node_pos_x,
                node_pos_y,
                node_pos_z,
                node_size,
                true,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const DEFAULT_NODE_SIZE: f32 = 4.0;
    const DEFAULT_POS_X: f32 = 1.0;
    const DEFAULT_POS_Y: f32 = 2.0;
    const DEFAULT_POS_Z: f32 = 3.0;
    const DEFAULT_OCTREE_WIDTH: u32 = 1;
    const DEFAULT_OCTREE_HEIGHT: u32 = 1;
    const DEFAULT_OCTREE_DEPTH: u32 = 1;

    #[test]
    fn test_new_octree() {
        let tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        assert_eq!(tree.get_root_node_size(), DEFAULT_NODE_SIZE);
        assert_eq!(tree.get_root_position(), (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        assert_eq!(tree.get_root_node_indices().len(), 1);
        let (root_index, x, y, z) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        assert_eq!((x, y, z), (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        assert_eq!(root_index, 0);
    }

    #[test]
    fn test_multiple_roots() {
        let x_multiplier = 2;
        let y_multiplier = 3;
        let z_multiplier = 4;
        let tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * x_multiplier, DEFAULT_OCTREE_HEIGHT * y_multiplier, DEFAULT_OCTREE_DEPTH * z_multiplier, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        for x in 0..x_multiplier {
            for y in 0..y_multiplier {
                for z in 0..z_multiplier {
                    let pos_x = DEFAULT_POS_X + x as f32 * DEFAULT_NODE_SIZE as f32;
                    let pos_y = DEFAULT_POS_Y + y as f32 * DEFAULT_NODE_SIZE as f32;
                    let pos_z = DEFAULT_POS_Z + z as f32 * DEFAULT_NODE_SIZE as f32;
                    let (root_index, x_pos, y_pos, z_pos) = tree.get_root_node_by_position(pos_x, pos_y, pos_z).expect("Root node should exist");
                    assert_eq!((x_pos, y_pos, z_pos), (pos_x, pos_y, pos_z));
                    assert_eq!(root_index, (x + y * x_multiplier + z * x_multiplier * y_multiplier) as usize);
                }
            }
        }
    }

    #[test]
    fn test_add_item_to_root_node() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // First index of the root node should contain the item
        assert_eq!(node.indices[0], 1);
        // All other indices should be empty
        for i in 1..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_add_item_to_subnode() {
        // Actually test all subnodes individually
        for n in 0..8 {
            let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
            let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
            // Make the item small enough to fit in a subnode
            let object_pos = (
                DEFAULT_POS_X + SUBNODE_OFFSETS[n].0 * DEFAULT_NODE_SIZE / 2.0 + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
                DEFAULT_POS_Y + SUBNODE_OFFSETS[n].1 * DEFAULT_NODE_SIZE / 2.0 + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
                DEFAULT_POS_Z + SUBNODE_OFFSETS[n].2 * DEFAULT_NODE_SIZE / 2.0 + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
            );
            tree.move_item(1, object_pos.0, object_pos.1, object_pos.2, DEFAULT_NODE_SIZE * 0.3, None);
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // The root node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The item should be in the one subnode
            assert_ne!(node.subnode_indices[n], u32::MAX);
            let subnode_index = node.subnode_indices[n] as usize;
            for i in 0..8 {
                if i != n {
                    assert_eq!(node.subnode_indices[i], u32::MAX);
                }
            }
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // Only the first index of the subnode should contain the item
            for i in 0..8 {
                if i != 0 {
                    assert_eq!(subnode.indices[i], u32::MAX);
                } else {
                    assert_eq!(subnode.indices[i], 1);
                }
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }
    }

    #[test]
    fn test_add_item_to_the_same_node_twice() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, None);
        { // Limit scope of "node"
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // First index of the root node should contain the item
            assert_eq!(node.indices[0], 1);
            // All other indices should be empty
            for i in 1..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The root note should not have any subnodes
            for i in 0..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
        }
        // Just do the same move operation again
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, Some((DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // First index of the root node should contain the item
        assert_eq!(node.indices[0], 1);
        // All other indices should be empty
        for i in 1..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_add_item_to_multiple_subnodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Add an item that is small enough to fit in subnodes, put it in the middle of the root node so it spans all of the subnodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Y + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Z + DEFAULT_NODE_SIZE / 2.0, DEFAULT_NODE_SIZE * 0.3, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The root node should not contain the item
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The item should be in all of the subnodes
        for i in 0..8 {
            assert_ne!(node.subnode_indices[i], u32::MAX);
            let subnode_index = node.subnode_indices[i] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }
    }

    #[test]
    fn test_move_item_to_other_subnode() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Make the item small enough to fit in a subnode
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        { // Putting this in a block to limit the scope of `node`
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // The root node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The item should be in the first subnode
            assert_ne!(node.subnode_indices[0], u32::MAX);
            let subnode_index = node.subnode_indices[0] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to a different subnode
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE / 2.0 + DEFAULT_NODE_SIZE * 0.3 / 2.0, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The item should be in the second subnode
        assert_ne!(node.subnode_indices[1], u32::MAX);
        let subnode_index = node.subnode_indices[1] as usize;
        let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(subnode.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(subnode.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_from_multiple_subnodes_to_other_multiple_subnodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Make the item small enough to fit in a subnode
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        { // Putting this in a block to limit the scope of `node`
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // The root node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The item should be in the third subnode
            assert_ne!(node.subnode_indices[2], u32::MAX);
            let subnode_index = node.subnode_indices[2] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to a different subnode
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE / 2.0 + DEFAULT_NODE_SIZE * 0.3 / 2.0, DEFAULT_POS_Y + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The item should be in the fourth subnode
        assert_ne!(node.subnode_indices[3], u32::MAX);
        let subnode_index = node.subnode_indices[3] as usize;
        let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(subnode.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(subnode.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_outside_of_octree() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Make the item small enough to fit in a subnode
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE * 2.0, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The root node should not contain the item
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The item should not be in any subnode
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_after_bigger_resize() {
        // First create an object in the root node
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, None);
        {
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // First index of the root node should contain the item
            assert_eq!(node.indices[0], 1);
            // All other indices should be empty
            for i in 1..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The root note should not have any subnodes
            for i in 0..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to the same position, but with a smaller size that fits in a subnode
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, Some((DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // Item should be removed from the root node
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The item should be in the first subnode
        assert_ne!(node.subnode_indices[0], u32::MAX);
        let subnode_index = node.subnode_indices[0] as usize;
        let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(subnode.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(subnode.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_after_smaller_resize() {
        // First create an object in a subnode
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        { // Putting this in a block to limit the scope of `node`
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // The root node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The item should be in the first subnode
            assert_ne!(node.subnode_indices[0], u32::MAX);
            let subnode_index = node.subnode_indices[0] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to the same position, but with a bigger size that no longer fits in a subnode
        // This should remove the item from the subnode and put it back in the root node
        tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, Some((DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // First index of the root node should contain the item
        assert_eq!(node.indices[0], 1);
        // All other indices should be empty
        for i in 1..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_out_of_empty_node() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Move an item to a position outside of the octree
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE * 2.0, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // The root node should not contain the item
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The item should not be in any subnode
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_between_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 2, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        { // Limiting the scope of `node`
            tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, None);
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // First index of the root node should contain the item
            assert_eq!(node.indices[0], 1);
            // All other indices should be empty
            for i in 1..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The root note should not have any subnodes
            for i in 0..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to the second root node
        let new_pos_x = DEFAULT_POS_X + DEFAULT_NODE_SIZE + DEFAULT_NODE_SIZE * 0.6 / 2.0;
        tree.move_item(1, new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.6, Some((DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let (root_index2, _, _, _) = tree.get_root_node_by_position(new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        let node2 = tree.nodes.get(root_index2).expect("Root node should exist");
        // First index of the root node should contain the item
        assert_eq!(node2.indices[0], 1);
        // All other indices should be empty
        for i in 1..8 {
            assert_eq!(node2.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node2.subnode_indices[i], u32::MAX);
        }

        // Check that the first root node no longer contains the item
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // All other indices should be empty
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_between_subnodes_in_different_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 2, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        { // Limiting the scope of `node`
            tree.move_item(1, DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // All indices should be empty
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The root note should have the item in the first subnode
            assert_ne!(node.subnode_indices[0], u32::MAX);
            // All other subnodes should be empty
            for i in 1..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
            let subnode_index = node.subnode_indices[0] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to a subnode in the second root node
        let new_pos_x = DEFAULT_POS_X + DEFAULT_NODE_SIZE + DEFAULT_NODE_SIZE * 0.3 / 2.0;
        tree.move_item(1, new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, Some((DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let (root_index2, _, _, _) = tree.get_root_node_by_position(new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        let node2 = tree.nodes.get(root_index2).expect("Root node should exist");
        // All indices should be empty
        for i in 0..8 {
            assert_eq!(node2.indices[i], u32::MAX);
        }
        // The root note should have the item in the first subnode
        assert_ne!(node2.subnode_indices[0], u32::MAX);
        // All other subnodes should be empty
        for i in 1..8 {
            assert_eq!(node2.subnode_indices[i], u32::MAX);
        }
        let subnode_index2 = node2.subnode_indices[0] as usize;
        let subnode2 = tree.nodes.get(subnode_index2).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(subnode2.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(subnode2.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode2.subnode_indices[i], u32::MAX);
        }

        // Check that the first root node no longer has any subnodes
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // All other indices should be empty
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_move_item_over_multiple_subnodes_between_different_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 2, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        { // Limiting the scope of `node`
            tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, None);
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // All indices should be empty
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The root note should have the item in the first 2 subnodes
            assert_ne!(node.subnode_indices[0], u32::MAX);
            assert_ne!(node.subnode_indices[1], u32::MAX);
            // All other subnodes should be empty
            for i in 2..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
            let subnode_index = node.subnode_indices[0] as usize;
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
            let other_subnode_index = node.subnode_indices[1] as usize;
            let other_subnode = tree.nodes.get(other_subnode_index).expect("Subnode should exist");
            // The subnode should contain the item
            assert_eq!(other_subnode.indices[0], 1);
            // All other indices in the subnode should be empty
            for i in 1..8 {
                assert_eq!(other_subnode.indices[i], u32::MAX);
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(other_subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move the item to a subnode in the second root node
        let new_pos_x = DEFAULT_POS_X + DEFAULT_NODE_SIZE * 1.5;
        tree.move_item(1, new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z, DEFAULT_NODE_SIZE * 0.3, Some((DEFAULT_POS_X + DEFAULT_NODE_SIZE / 2.0, DEFAULT_POS_Y, DEFAULT_POS_Z)));
        let (root_index2, _, _, _) = tree.get_root_node_by_position(new_pos_x, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        let node2 = tree.nodes.get(root_index2).expect("Root node should exist");
        // All indices should be empty
        for i in 0..8 {
            assert_eq!(node2.indices[i], u32::MAX);
        }
        // The root note should have the item in the first 2 subnodes
        assert_ne!(node2.subnode_indices[0], u32::MAX);
        assert_ne!(node2.subnode_indices[1], u32::MAX);
        // All other subnodes should be empty
        for i in 2..8 {
            assert_eq!(node2.subnode_indices[i], u32::MAX);
        }
        let subnode_index2 = node2.subnode_indices[0] as usize;
        let subnode2 = tree.nodes.get(subnode_index2).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(subnode2.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(subnode2.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode2.subnode_indices[i], u32::MAX);
        }
        let other_subnode_index2 = node2.subnode_indices[1] as usize;
        let other_subnode2 = tree.nodes.get(other_subnode_index2).expect("Subnode should exist");
        // The subnode should contain the item
        assert_eq!(other_subnode2.indices[0], 1);
        // All other indices in the subnode should be empty
        for i in 1..8 {
            assert_eq!(other_subnode2.indices[i], u32::MAX);
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(other_subnode2.subnode_indices[i], u32::MAX);
        }

        // Check that the first root node no longer has any subnodes
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // All other indices should be empty
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The root note should not have any subnodes
        for i in 0..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_item_in_multiple_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 2, DEFAULT_OCTREE_HEIGHT * 2, DEFAULT_OCTREE_DEPTH * 2, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        // Put item in the middle of the chunks so it overlaps with all 8 root nodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.6, None);
        for root_node_index in tree.get_root_node_indices() {
            let node = tree.nodes.get(root_node_index.clone() as usize).expect("Root node should exist");
            // node should contain the item
            assert_eq!(node.indices[0], 1);
            // All other indices in the node should be empty
            for i in 1..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The node should not have any further subnodes
            for i in 0..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
        }
    }

    #[test]
    fn test_item_in_multiple_subnodes_of_different_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 2, DEFAULT_OCTREE_HEIGHT * 2, DEFAULT_OCTREE_DEPTH * 2, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        // Put item in the middle of the chunks so it overlaps with all 8 root nodes
        // Make it small enough so it's one subnode layer down
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.3, None);
        for root_node_index in tree.get_root_node_indices() {
            let node = tree.nodes.get(root_node_index.clone() as usize).expect("Root node should exist");
            // node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // Find the subnode that exists and will thus contain the item
            let mut found_subnode_index = u32::MAX;
            for i in 0..8 {
                if node.subnode_indices[i] != u32::MAX {
                    // There should only be 1 subnode that contains the item
                    assert_eq!(found_subnode_index, u32::MAX);
                    found_subnode_index = node.subnode_indices[i];
                }
            }
            // We should have a subnode index found
            assert_ne!(found_subnode_index, u32::MAX);
            let subnode = tree.nodes.get(found_subnode_index as usize).expect("Node should exist");
            // node should contain the item
            assert_eq!(subnode.indices[0], 1);
            // All other indices in the node should be empty
            for i in 1..8 {
                assert_eq!(subnode.indices[i], u32::MAX);
            }
            // The node should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }
    }

    #[test]
    fn test_move_item_from_multiple_root_nodes_to_other_multiple_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 3, DEFAULT_OCTREE_HEIGHT * 2, DEFAULT_OCTREE_DEPTH * 2, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        // Put item in the middle of the chunks so it overlaps with all 8 root nodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.6, None);
        for x in 0..1 {
            for y in 0..1 {
                for z in 0..1 {
                    let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X + DEFAULT_NODE_SIZE * x as f32, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                    let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                    // node should contain the item
                    assert_eq!(node.indices[0], 1);
                    // All other indices in the node should be empty
                    for i in 1..8 {
                        assert_eq!(node.indices[i], u32::MAX);
                    }
                    // The node should not have any further subnodes
                    for i in 0..8 {
                        assert_eq!(node.subnode_indices[i], u32::MAX);
                    }
                }
            }
        }
        // Now move the item so it overlaps with a different set of root nodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE * 2.0, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.6, Some((DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE)));
        for x in 1..2 {
            for y in 0..1 {
                for z in 0..1 {
                    let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X + DEFAULT_NODE_SIZE * x as f32, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                    let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                    // node should contain the item
                    assert_eq!(node.indices[0], 1);
                    // All other indices in the node should be empty
                    for i in 1..8 {
                        assert_eq!(node.indices[i], u32::MAX);
                    }
                    // The node should not have any further subnodes
                    for i in 0..8 {
                        assert_eq!(node.subnode_indices[i], u32::MAX);
                    }
                }
            }
        }
        // Check that the other root nodes no longer contain the item
        for y in 0..1 {
            for z in 0..1 {
                let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                // All indices in the node should be empty
                for i in 0..8 {
                    assert_eq!(node.indices[i], u32::MAX);
                }
                // The node should not have any further subnodes
                for i in 0..8 {
                    assert_eq!(node.subnode_indices[i], u32::MAX);
                }
            }
        }
    }

    #[test]
    fn test_move_item_from_multiple_subnodes_in_root_nodes_to_other_multiple_subnodes_in_root_nodes() {
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * 3, DEFAULT_OCTREE_HEIGHT * 2, DEFAULT_OCTREE_DEPTH * 2, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        // Put item in the middle of the chunks so it overlaps with all 8 root nodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.3, None);
        for x in 0..1 {
            for y in 0..1 {
                for z in 0..1 {
                    let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X + DEFAULT_NODE_SIZE * x as f32, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                    let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                    // node should not contain the item
                    for i in 0..8 {
                        assert_eq!(node.indices[i], u32::MAX);
                    }
                    // Find the subnode that exists and will thus contain the item
                    let mut found_subnode_index = u32::MAX;
                    for i in 0..8 {
                        if node.subnode_indices[i] != u32::MAX {
                            // There should only be 1 subnode that contains the item
                            assert_eq!(found_subnode_index, u32::MAX);
                            found_subnode_index = node.subnode_indices[i];
                        }
                    }
                    // We should have a subnode index found
                    assert_ne!(found_subnode_index, u32::MAX);
                    let subnode = tree.nodes.get(found_subnode_index as usize).expect("Node should exist");
                    // node should contain the item
                    assert_eq!(subnode.indices[0], 1);
                    // All other indices in the node should be empty
                    for i in 1..8 {
                        assert_eq!(subnode.indices[i], u32::MAX);
                    }
                    // The node should not have any further subnodes
                    for i in 0..8 {
                        assert_eq!(subnode.subnode_indices[i], u32::MAX);
                    }
                }
            }
        }
        // Now move the item so it overlaps with a different set of root nodes
        tree.move_item(1, DEFAULT_POS_X + DEFAULT_NODE_SIZE * 2.0, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE, DEFAULT_NODE_SIZE * 0.3, Some((DEFAULT_POS_X + DEFAULT_NODE_SIZE, DEFAULT_POS_Y + DEFAULT_NODE_SIZE, DEFAULT_POS_Z + DEFAULT_NODE_SIZE)));
        for x in 1..2 {
            for y in 0..1 {
                for z in 0..1 {
                    let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X + DEFAULT_NODE_SIZE * x as f32, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                    let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                    // node should not contain the item
                    for i in 1..8 {
                        assert_eq!(node.indices[i], u32::MAX);
                    }
                    // Find the subnode that exists and will thus contain the item
                    let mut found_subnode_index = u32::MAX;
                    for i in 0..8 {
                        if node.subnode_indices[i] != u32::MAX {
                            // There should only be 1 subnode that contains the item
                            assert_eq!(found_subnode_index, u32::MAX);
                            found_subnode_index = node.subnode_indices[i];
                        }
                    }
                    // We should have a subnode index found
                    assert_ne!(found_subnode_index, u32::MAX);
                    let subnode = tree.nodes.get(found_subnode_index as usize).expect("Node should exist");
                    // node should contain the item
                    assert_eq!(subnode.indices[0], 1);
                    // All other indices in the node should be empty
                    for i in 1..8 {
                        assert_eq!(subnode.indices[i], u32::MAX);
                    }
                    // The node should not have any further subnodes
                    for i in 0..8 {
                        assert_eq!(subnode.subnode_indices[i], u32::MAX);
                    }
                }
            }
        }
        // Check that the other root nodes no longer contain the item
        for y in 0..1 {
            for z in 0..1 {
                let root_node_index = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y + DEFAULT_NODE_SIZE * y as f32, DEFAULT_POS_Z + DEFAULT_NODE_SIZE * z as f32).expect("Root node index should be found").0;
                let node = tree.nodes.get(root_node_index as usize).expect("Root node should exist");
                // All indices in the node should be empty
                for i in 0..8 {
                    assert_eq!(node.indices[i], u32::MAX);
                }
                // The node should not have any further subnodes
                for i in 0..8 {
                    assert_eq!(node.subnode_indices[i], u32::MAX);
                }
            }
        }
    }

    #[test]
    fn test_remove_non_existent_item() {
        // Create a tree with 1 item in it just to construct a subnode
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH, DEFAULT_OCTREE_HEIGHT, DEFAULT_OCTREE_DEPTH, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        let (root_index, _, _, _) = tree.get_root_node_by_position(DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z).expect("Root node should exist");
        // Make the item small enough to fit in a subnode
        let object_pos = (
            DEFAULT_POS_X + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
            DEFAULT_POS_Y + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
            DEFAULT_POS_Z + DEFAULT_NODE_SIZE * 0.3 / 2.0 + 0.1,
        );
        tree.move_item(1, object_pos.0, object_pos.1, object_pos.2, DEFAULT_NODE_SIZE * 0.3, None);
        { // Limit scope of "node"
            let node = tree.nodes.get(root_index).expect("Root node should exist");
            // The root node should not contain the item
            for i in 0..8 {
                assert_eq!(node.indices[i], u32::MAX);
            }
            // The item should be in the first subnode
            assert_ne!(node.subnode_indices[0], u32::MAX);
            let subnode_index = node.subnode_indices[0] as usize;
            for i in 1..8 {
                assert_eq!(node.subnode_indices[i], u32::MAX);
            }
            let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
            // Only the first index of the subnode should contain the item
            for i in 0..8 {
                if i != 0 {
                    assert_eq!(subnode.indices[i], u32::MAX);
                } else {
                    assert_eq!(subnode.indices[i], 1);
                }
            }
            // The subnode should not have any further subnodes
            for i in 0..8 {
                assert_eq!(subnode.subnode_indices[i], u32::MAX);
            }
        }

        // Now move an item with a different id, but the same position out of that node
        tree.move_item(2, f32::MAX, f32::MAX, f32::MAX, DEFAULT_NODE_SIZE * 0.3, Some((object_pos.0, object_pos.1, object_pos.2)));
        let node = tree.nodes.get(root_index).expect("Root node should exist");
        // Exactly the same checks should still apply
        // The root node should not contain the item
        for i in 0..8 {
            assert_eq!(node.indices[i], u32::MAX);
        }
        // The item should be in the first subnode
        assert_ne!(node.subnode_indices[0], u32::MAX);
        let subnode_index = node.subnode_indices[0] as usize;
        for i in 1..8 {
            assert_eq!(node.subnode_indices[i], u32::MAX);
        }
        let subnode = tree.nodes.get(subnode_index).expect("Subnode should exist");
        // Only the first index of the subnode should contain the item
        for i in 0..8 {
            if i != 0 {
                assert_eq!(subnode.indices[i], u32::MAX);
            } else {
                assert_eq!(subnode.indices[i], 1);
            }
        }
        // The subnode should not have any further subnodes
        for i in 0..8 {
            assert_eq!(subnode.subnode_indices[i], u32::MAX);
        }
    }

    #[test]
    fn test_insert_tons_of_items() {
        // Test that it can handle inserting a ton of items and that it allocates the expected amount of memory
        const ROOT_NODES: u32 = 3;
        const LEVELS: u32 = 5;
        const SUBNODE_SIZE: f32 = DEFAULT_NODE_SIZE / ((2 as i32).pow(LEVELS) as f32);
        const OBJECT_SIZE: f32 = SUBNODE_SIZE * 0.6;
        const ONE_DIMENSION: i32 = (2 as i32).pow(LEVELS) * ROOT_NODES as i32;
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * ROOT_NODES, DEFAULT_OCTREE_HEIGHT * ROOT_NODES, DEFAULT_OCTREE_DEPTH * ROOT_NODES, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        for x in 0..ONE_DIMENSION {
            for y in 0..ONE_DIMENSION {
                for z in 0..ONE_DIMENSION {
                    let object_pos = (
                        DEFAULT_POS_X + ((x as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Y + ((y as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Z + ((z as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                    );
                    let item_id = 
                        z + 
                        y * ONE_DIMENSION +
                        x * ONE_DIMENSION * ONE_DIMENSION;
                    tree.move_item(item_id as u32,
                    object_pos.0,
                    object_pos.1,
                    object_pos.2,
                    OBJECT_SIZE,
                    None);
                }
            }
        }
        // Every subnode at the marked level should be filled, so it should be easy to calculate the exact amount of nodes in the structure
        let mut expected_size = ROOT_NODES.pow(3);
        for i in 1..LEVELS + 1 {
            expected_size += (8 as i32).pow(i) as u32 * ROOT_NODES.pow(3);
        }
        assert_eq!(expected_size, tree.nodes.len() as u32);
    }

    #[test]
    fn test_remove_tons_of_items() {
        // Test that removing a ton of items frees up the nodes
        const ROOT_NODES: u32 = 4;
        const LEVELS: u32 = 4;
        const SUBNODE_SIZE: f32 = DEFAULT_NODE_SIZE / ((2 as i32).pow(LEVELS) as f32);
        const OBJECT_SIZE: f32 = SUBNODE_SIZE * 0.6;
        const ONE_DIMENSION: i32 = (2 as i32).pow(LEVELS) * ROOT_NODES as i32;
        let mut tree = Octree::new(DEFAULT_NODE_SIZE, DEFAULT_OCTREE_WIDTH * ROOT_NODES, DEFAULT_OCTREE_HEIGHT * ROOT_NODES, DEFAULT_OCTREE_DEPTH * ROOT_NODES, (DEFAULT_POS_X, DEFAULT_POS_Y, DEFAULT_POS_Z));
        for x in 0..ONE_DIMENSION {
            for y in 0..ONE_DIMENSION {
                for z in 0..ONE_DIMENSION {
                    let object_pos = (
                        DEFAULT_POS_X + ((x as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Y + ((y as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Z + ((z as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                    );
                    let item_id = 
                        z + 
                        y * ONE_DIMENSION +
                        x * ONE_DIMENSION * ONE_DIMENSION;
                    tree.move_item(item_id as u32,
                    object_pos.0,
                    object_pos.1,
                    object_pos.2,
                    OBJECT_SIZE,
                    None);
                }
            }
        }
        // Every subnode at the marked level should be filled, so it should be easy to calculate the exact amount of nodes in the structure
        let mut expected_full_size = ROOT_NODES.pow(3);
        for i in 1..LEVELS + 1 {
            expected_full_size += (8 as i32).pow(i) as u32 * ROOT_NODES.pow(3);
        }
        assert_eq!(expected_full_size, tree.nodes.len() as u32);

        // Now move out all of the objects, which should bring the tree back to essentially its smallest form with only root nodes
        for x in 0..ONE_DIMENSION {
            for y in 0..ONE_DIMENSION {
                for z in 0..ONE_DIMENSION {
                    let object_pos = (
                        DEFAULT_POS_X + ((x as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Y + ((y as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                        DEFAULT_POS_Z + ((z as f32) * SUBNODE_SIZE) + (SUBNODE_SIZE / 2.0),
                    );
                    let item_id = 
                        z + 
                        y * ONE_DIMENSION +
                        x * ONE_DIMENSION * ONE_DIMENSION;
                    tree.move_item(item_id as u32,
                    f32::MAX,
                    f32::MAX,
                    f32::MAX,
                    OBJECT_SIZE,
                    Some(object_pos));
                }
            }
        }
        // Only root nodes should be left
        assert_eq!(ROOT_NODES.pow(3), tree.nodes.len() as u32);
        // The actual size of the nodes list should actually remain the same, just empty nodes
        assert_eq!(expected_full_size, tree.nodes.full_len() as u32);

        // Let's add 1 item back and check it takes up the expected amount of memory
        tree.move_item(
            1u32,
            DEFAULT_POS_X,
            DEFAULT_POS_Y,
            DEFAULT_POS_Z,
            OBJECT_SIZE,
            None);
        // Now the nodes should be filled for all root nodes and 1 node per level
        assert_eq!(ROOT_NODES.pow(3) + LEVELS, tree.nodes.len() as u32);
        // The actual total memory usage of the nodes should remain unchanged
        assert_eq!(expected_full_size, tree.nodes.full_len() as u32);
    }
}