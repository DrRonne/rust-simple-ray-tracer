// A struct containing a list where the index of an added item is fixed
// For example, in a regular list, removing item at index 0 shifts all subsequent items down by one.
// In a fixed index list, removing an item does not change the index of the remaining items.
// Instead of shifting items, it simply marks the index as empty.
// Adding new items will fill the first empty index available. Otherwise, it will append to the end of the list.
// Subsequently, the list can only ever really grow, and never shrink. Proceed with caution.
pub struct TombstonedList<T> {
    items: Vec<T>,
    empty_indices: Vec<usize>,
    default_value: T,
}

impl<T: Clone> TombstonedList<T> {
    pub fn new(default_value: T) -> Self {
        Self {
            items: Vec::new(),
            empty_indices: Vec::new(),
            default_value,
        }
    }

    pub fn add(&mut self, item: T) -> usize {
        if let Some(index) = self.empty_indices.pop() {
            self.items[index] = item;
            index
        } else {
            self.items.push(item);
            self.items.len() - 1
        }
    }

    pub fn remove(&mut self, index: usize) {
        if index < self.items.len() {
            self.empty_indices.push(index);
            self.items[index] = self.default_value.clone();
        }
    }

    // Unchecked!
    // Make sure the index is valid before calling this method.
    pub fn update(&mut self, index: usize, item: T) {
        if index < self.items.len() {
            self.items[index] = item;
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)
    }

    pub fn len(&self) -> usize {
        // Size of the thombstoned list, excluding "empty" fields
        self.items.len() - self.empty_indices.len()
    }

    pub fn full_len(&self) -> usize {
        // Size of the thombstoned list, including "empty" fields
        self.items.len()
    }

    pub fn get_items(&self) -> &Vec<T> {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_list_is_empty() {
        let list = TombstonedList::new(u32::MAX);
        assert_eq!(list.len(), 0);
        assert_eq!(list.full_len(), 0);
    }

    #[test]
    fn test_add_and_get_item() {
        let mut list = TombstonedList::new(u32::MAX);
        let index = list.add(1);
        assert_eq!(list.get(index), Some(&1));
    }

    #[test]
    fn test_add_and_get_mut_item() {
        let mut list = TombstonedList::new(u32::MAX);
        let index = list.add(1);
        assert_eq!(list.get_mut(index), Some(&mut 1));
    }

    #[test]
    fn test_remove_item() {
        let mut list = TombstonedList::new(u32::MAX);
        let index = list.add(1);
        list.remove(index);
        assert_eq!(list.get(index), Some(&u32::MAX));
        assert_eq!(list.len(), 0);
        assert_eq!(list.full_len(), 1);
    }

    #[test]
    fn test_get_items() {
        let mut list = TombstonedList::new(u32::MAX);
        list.add(1);
        list.add(2);
        assert_eq!(list.get_items(), &vec![1, 2]);
    }

    #[test]
    fn test_add_to_empty_index() {
        let mut list = TombstonedList::new(u32::MAX);
        let index1 = list.add(1);
        list.add(2);
        list.remove(index1);
        let index3 = list.add(3);
        assert_eq!(index3, index1); // Should reuse the empty index
        assert_eq!(list.get(index3), Some(&3));
        assert_eq!(list.len(), 2); // 2 items should be present
        assert_eq!(list.full_len(), 2);
    }

    #[test]
    fn get_items_after_removal() {
        let mut list = TombstonedList::new(u32::MAX);
        list.add(1);
        let index = list.add(2);
        list.remove(index);
        assert_eq!(list.get_items(), &vec![1, u32::MAX]);
    }

    #[test]
    fn test_update_item() {
        let mut list = TombstonedList::new(u32::MAX);
        let index = list.add(1);
        list.add(2); // Add another item to ensure the list has more than one item
        list.update(index, 2);
        assert_eq!(list.get(index), Some(&2));
    }

    #[test]
    fn test_update_item_out_of_bounds() {
        let mut list = TombstonedList::new(u32::MAX);
        list.add(1);
        list.update(10, 2); // Should not panic, but do nothing
        assert_eq!(list.get(0), Some(&1));
    }

    #[test]
    fn get_items_after_update() {
        let mut list = TombstonedList::new(u32::MAX);
        list.add(1);
        let index = list.add(2);
        list.update(index, 3);
        assert_eq!(list.get_items(), &vec![1, 3]);
    }

    #[test]
    fn get_items_after_multiple_removals_and_additions_and_updates() {
        let mut list = TombstonedList::new(u32::MAX);
        let index1 = list.add(1);
        let index2 = list.add(2);
        list.remove(index1);
        let index3 = list.add(3);
        list.update(index2, 4);
        list.update(index3, 5);
        assert_eq!(index3, index1); // Should reuse the empty index
        assert_eq!(list.get_items(), &vec![5, 4]);
    }
}