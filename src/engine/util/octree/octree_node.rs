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
