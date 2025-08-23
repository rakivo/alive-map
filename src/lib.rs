#![doc = include_str!("../README.md")]

#![no_std]

#![warn(
    anonymous_parameters,
    missing_copy_implementations,
    missing_debug_implementations,
    nonstandard_style,
    rust_2018_idioms,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_extern_crates,
    unused_qualifications,
    variant_size_differences
)]

extern crate alloc;

use core::{fmt, mem};
use core::ops::{Deref, DerefMut};
use core::hash::{Hash, BuildHasher};
use core::iter::{FromIterator, FusedIterator};

use alloc::vec::Vec;

use hashbrown::{HashMap, DefaultHashBuilder};

/// A key-value entry in an `AliveMap`.
///
/// Contains the key, optional value, and a flag indicating if the entry is alive.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: Option<V>,

    // doubly linked list pointers (indexes into entries vec)
    prev: Option<usize>,
    next: Option<usize>
}

impl<K, V> Deref for Entry<K, V> {
    type Target = Option<V>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<K, V> DerefMut for Entry<K, V> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<K, V> Clone for Entry<K, V>
where
    K: Clone,
    V: Clone
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            prev: self.prev,
            next: self.next,
            key: self.key.clone(),
            value: self.value.clone(),
        }
    }
}

impl<K, V> Entry<K, V> {
    #[inline(always)]
    const fn new(key: K, value: V) -> Self {
        let value = Some(value);
        Self { key, value, prev: None, next: None }
    }

    #[inline(always)]
    const fn is_alive(&self) -> bool {
        self.value.is_some()
    }
}

/// An insertion-order-preserving hash map that tracks alive entries.
///
/// Entries can be removed (marked dead) and later resurrected, and the map
/// can be compacted to remove dead slots while preserving order.
pub struct AliveMap<K, V, S = DefaultHashBuilder> {
    index_map: HashMap<K, usize, S>, // key -> index in entries
    entries: Vec<Entry<K, V>>, // all entries in insertion order

    // doubly linked list of alive entries
    head: Option<usize>, // first alive entry
    tail: Option<usize>, // last alive entry

    alive_count: usize,
}

impl<K, V> AliveMap<K, V, DefaultHashBuilder> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initially created with a capacity of 0, so it will not allocate until it
    /// is first inserted into.
    #[inline]
    pub fn new() -> Self {
        Self {
            index_map: HashMap::new(),
            head: None,
            tail: None,
            entries: Vec::new(),
            alive_count: 0,
        }
    }

    /// Creates an empty `HashMap` with the specified capacity.
    ///
    /// The hash map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the hash map will not allocate.
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            head: None,
            tail: None,
            index_map: HashMap::with_capacity(n),
            entries: Vec::with_capacity(n),
            alive_count: 0,
        }
    }
}

impl<K, V, S> AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    S: Default + BuildHasher,
{
    /// Creates an empty `AliveMap` using the provided hasher `h`.
    #[inline]
    pub fn with_hasher(h: S) -> Self {
        Self::with_capacity_and_hasher(0, h)
    }

    /// Creates an empty `AliveMap` with the specified initial capacity `n` and hasher `h`.
    ///
    /// The map will be able to hold at least `n` elements without reallocating.
    #[inline]
    pub fn with_capacity_and_hasher(n: usize, h: S) -> Self {
        Self {
            index_map: HashMap::with_capacity_and_hasher(n, h),
            entries: Vec::with_capacity(n),
            alive_count: 0,
            head: None,
            tail: None
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists and is alive, its value is updated.
    /// If the key exists but is marked dead, it is resurrected and its value updated.
    ///
    /// # Examples
    ///
    /// ```
    /// use alive_map::AliveMap;
    ///
    /// let mut map = AliveMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.insert(1, "b"), Some("a"));
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&idx) = self.index_map.get(&key) {
            let entry = &mut self.entries[idx];

            if let Some(old_value) = entry.take() {
                entry.value = Some(value);
                self.unlink(idx);
                self.link_tail(idx);
                return Some(old_value)
            } else {
                entry.value = Some(value);
                self.alive_count += 1;
                self.link_tail(idx);
                return None
            }
        }

        // append to the end
        let idx = self.entries.len();
        self.entries.push(Entry::new(key.clone(), value));
        self.index_map.insert(key, idx);
        self.link_tail(idx);
        self.alive_count += 1;
        None
    }

    /// Removes a key from the map, returning its value if it was alive.
    ///
    /// The slot is marked dead but not removed from internal storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use alive_map::AliveMap;
    ///
    /// let mut map = AliveMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.get(&1), None);
    /// ```
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = *self.index_map.get(key)?;

        let value = self.entries[idx].take()?;

        self.unlink(idx);
        self.alive_count -= 1;

        Some(value)
    }

    /// Returns a reference to the value corresponding to the key, if it is alive.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.index_map.get(key).and_then(|&idx| {
            let entry = &self.entries[idx];
            if entry.is_alive() {
                entry.value.as_ref()
            } else {
                None
            }
        })
    }

    /// Returns `true` if the map contains a value for the specified key.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Returns a mutable reference to the value corresponding to the key, if it is alive.
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.index_map.get(key).and_then(|&idx| {
            let entry = &mut self.entries[idx];
            if entry.is_alive() {
                entry.value.as_mut()
            } else {
                None
            }
        })
    }

    /// Returns the number of alive entries in the map.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.alive_count
    }

    /// Returns `true` if the map contains no alive entries.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.alive_count == 0
    }

    /// # Examples
    ///
    /// ```
    /// use alive_map::AliveMap;
    ///
    /// let mut map = AliveMap::new();
    ///
    /// // Insert entries
    /// map.insert(1, "a");
    /// map.insert(2, "b");
    /// map.insert(3, "c");
    /// assert_eq!(map.len(), 3); // three alive entries
    ///
    /// // Remove an entry
    /// assert_eq!(map.remove(&2), Some("b")); // remove key 2
    /// assert_eq!(map.get(&2), None);        // key 2 is dead
    /// assert_eq!(map.len(), 2);             // only two alive entries remain
    ///
    /// // Iterate over alive entries in insertion order
    /// let items: Vec<_> = map.iter().collect();
    /// assert_eq!(items, vec![(&1, &"a"), (&3, &"c")]);
    ///
    /// // Resurrect a dead entry by reinserting
    /// map.insert(2, "B"); // key 2 resurrected with new value
    /// assert_eq!(map.len(), 3); // three alive entries again
    ///
    /// // Iteration still preserves insertion order for alive entries
    /// let items: Vec<_> = map.iter().collect();
    /// assert_eq!(items, vec![(&1, &"a"), (&3, &"c"), (&2, &"B")]);
    ///
    /// // Compact internal storage, removing dead slots
    /// map.shrink_to_fit_alive();
    /// assert_eq!(map.len(), 3); // alive count unchanged
    /// ```
    pub fn shrink_to_fit_alive(&mut self)
    where
        K: Clone,
        V: Clone
    {
        if self.alive_count == self.entries.len() {
            self.index_map.shrink_to_fit();
            self.entries.shrink_to_fit();
            return
        }

        let old_head = self.head;
        let old_entries = mem::take(&mut self.entries);

        let mut new_entries = Vec::with_capacity(self.alive_count);

        self.index_map.clear();
        self.head = None;
        self.tail = None;

        let mut curr = old_head;
        while let Some(old_idx) = curr {
            let old_entry = &old_entries[old_idx];
            if !old_entry.is_alive() {
                curr = old_entry.next;
                continue
            }

            let new_idx = new_entries.len();

            let prev_idx = if new_entries.is_empty() {
                None
            } else {
                Some(new_idx - 1)
            };

            new_entries.push(Entry {
                key: old_entry.key.clone(),
                value: old_entry.value.clone(),
                prev: prev_idx,
                next: None,
            });

            if let Some(prev) = prev_idx {
                new_entries[prev].next = Some(new_idx)
            }

            // update index map
            self.index_map.insert(old_entry.key.clone(), new_idx);

            if self.head.is_none() {
                self.head = Some(new_idx)
            }

            self.tail = Some(new_idx);

            curr = old_entry.next;
        }

        self.entries = new_entries;
    }

    /// Reserves capacity for at least `additional` more alive entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use alive_map::AliveMap;
    ///
    /// let mut map = AliveMap::<i32, i32>::new();
    /// map.reserve(100);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.index_map.reserve(additional);
        self.entries.reserve(additional);
    }

    /// Returns an iterator over alive entries in insertion order.
    ///
    /// The iterator implements `ExactSizeIterator` and `FusedIterator`.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            entries: &self.entries,
            curr: self.head,
            remaining: self.alive_count,
        }
    }

    /// Removes an entry from the internal linked list
    ///
    /// Marked as public just in case
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn unlink(&mut self, idx: usize) {
        let entry = &mut self.entries[idx];

        let prev = entry.prev;
        let next = entry.next;

        entry.prev = None;
        entry.next = None;

        if let Some(prev_idx) = prev {
            self.entries[prev_idx].next = next
        } else {
            // this was the head
            self.head = next
        }

        if let Some(next_idx) = next {
            self.entries[next_idx].prev = prev
        } else {
            // this was the tail
            self.tail = prev
        }
    }

    /// Adds an entry to the end of the internal linked list.
    ///
    /// Marked as public just in case
    #[cfg_attr(feature = "inline-more", inline)]
    pub fn link_tail(&mut self, idx: usize) {
        let entry = &mut self.entries[idx];
        entry.prev = self.tail;
        entry.next = None;

        if let Some(tail_idx) = self.tail {
            self.entries[tail_idx].next = Some(idx)
        } else {
            self.head = Some(idx)
        }

        self.tail = Some(idx)
    }
}

/// Borrowing iterator over alive entries in insertion order.
#[derive(Debug)]
pub struct Iter<'a, K, V> {
    entries: &'a Vec<Entry<K, V>>,
    curr: Option<usize>,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[cfg_attr(feature = "inline-more", inline)]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.curr {
            let entry = &self.entries[idx];
            self.curr = entry.next;
            self.remaining -= 1;
            Some((&entry.key, entry.value.as_ref().unwrap()))
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

impl<K, V, S> FromIterator<(K, V)> for AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    S: Default + BuildHasher,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut map = AliveMap::with_capacity_and_hasher(
            iter.size_hint().0,
            S::default()
        );
        iter.for_each(|(k, v)| _ = map.insert(k, v));
        map
    }
}

impl<K, V, S> Extend<(K, V)> for AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    S: Default + BuildHasher,
{
    #[cfg_attr(feature = "inline-more", inline)]
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };
        self.reserve(reserve);
        iter.for_each(move |(k, v)| _ = self.insert(k, v));
    }
}

impl<K, V, S> Default for AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    S: Default + BuildHasher,
{
    #[inline]
    fn default() -> Self {
        Self::with_capacity_and_hasher(0, S::default())
    }
}

impl<K, V> Clone for AliveMap<K, V, DefaultHashBuilder>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        Self {
            head: self.head,
            tail: self.tail,
            index_map: self.index_map.clone(),
            entries: self.entries.clone(),
            alive_count: self.alive_count
        }
    }
}

impl<K, V, S> PartialEq for AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    V: PartialEq,
    S: Default + BuildHasher,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().eq(other.iter())
    }
}

impl<K, V, S> Eq for AliveMap<K, V, S>
where
    K: Eq + Hash + Clone,
    V: Eq,
    S: Default + BuildHasher,
{
}

impl<K, V, S> fmt::Debug for AliveMap<K, V, S>
where
    K: fmt::Debug + Eq + Hash + Clone,
    V: fmt::Debug,
    S: Default + BuildHasher,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    use alloc::vec;
    use alloc::string::String;

    #[test]
    fn test_new_and_default_and_with_capacity() {
        let a: AliveMap<u64, u64> = AliveMap::new();
        assert!(a.is_empty());
        let b: AliveMap<u64, u64> = AliveMap::default();
        assert!(b.is_empty());
        let c: AliveMap<u64, u64> = AliveMap::with_capacity(10);
        assert!(c.is_empty());
        assert!(c.entries.capacity() >= 10);
    }

    #[test]
    fn test_insert_get_remove_iter_basic() {
        let mut sm = AliveMap::new();
        sm.insert(42u64, "foo");
        sm.insert(7u64, "bar");
        sm.insert(99u64, "baz");

        assert_eq!(sm.len(), 3);
        assert_eq!(sm.get(&42), Some(&"foo"));
        assert_eq!(sm.remove(&7), Some("bar"));
        assert_eq!(sm.get(&7), None);
        assert_eq!(sm.len(), 2);

        let items: Vec<_> = sm.iter().collect();
        assert_eq!(items, vec![(&42, &"foo"), (&99, &"baz")]);
    }

    #[test]
    fn test_overwrite_and_no_duplicate() {
        let mut sm = AliveMap::new();
        sm.insert(1, "a");
        sm.insert(1, "b"); // overwrite same key
        assert_eq!(sm.entries.len(), 1); // no duplicate slot
        assert_eq!(sm.len(), 1);
        assert_eq!(sm.get(&1), Some(&"b"));
    }

    #[test]
    fn test_resurrect_preserves_position() {
        let mut sm = AliveMap::new();
        sm.insert(1, "a");
        sm.insert(2, "b");
        assert_eq!(sm.entries.len(), 2);
        assert_eq!(sm.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![1, 2]);

        // remove 1 then resurrect
        assert_eq!(sm.remove(&1), Some("a"));
        assert_eq!(sm.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![2]);
        sm.insert(1, "c"); // resurrect, should keep original slot
        assert_eq!(sm.entries.len(), 2); // still 2
        // Now iteration yields 1 then 2 because original insertion order is preserved for alive entries
        assert_eq!(
            sm.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            vec![(2, "b"), (1, "c")]
        );
    }

    #[test]
    fn test_get_mut_changes_value() {
        let mut sm = AliveMap::new();
        sm.insert(10, String::from("hello"));
        {
            let s = sm.get_mut(&10).unwrap();
            s.push_str("_world");
        }
        assert_eq!(sm.get(&10).map(|s| s.as_str()), Some("hello_world"));
    }

    #[test]
    fn test_from_iterator_and_extend() {
        let src = vec![(1u32, "a"), (2, "b"), (3, "c")];
        let map: AliveMap<_, _> = src.clone().into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(
            map.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            src
        );

        let mut m2 = AliveMap::new();
        m2.extend(src.clone());
        assert_eq!(m2.len(), 3);
        assert_eq!(
            m2.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            src
        );
    }

    #[test]
    fn test_clone_and_partial_eq_and_eq() {
        let mut a = AliveMap::new();
        a.insert(1u8, 10u8);
        a.insert(2, 20);
        let b = a.clone();
        assert_eq!(a, b);

        // remove one and ensure inequality
        let mut c = b.clone();
        c.remove(&1);
        assert_ne!(a, c);

        // equality only considers alive entries in insertion order
        let mut d = AliveMap::new();
        d.insert(1u8, 10u8);
        d.insert(2, 20);
        assert_eq!(a, d);
    }

    #[test]
    fn test_debug_contains_keys() {
        let mut a = AliveMap::new();
        a.insert(5, "five");
        a.insert(6, "six");
        let s = format!("{:?}", a);
        assert!(s.contains("5"));
        assert!(s.contains("six"));
    }

    #[test]
    fn test_reserve_affects_capacity() {
        let mut m: AliveMap<u64, u64> = AliveMap::new();
        let _before_entries = m.entries.capacity();
        m.reserve(100);
        assert!(m.entries.capacity() >= 100);
        assert!(m.index_map.capacity() >= 100);
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_none() {
        let mut m: AliveMap<u32, u32> = AliveMap::new();
        assert_eq!(m.remove(&123), None);
    }

    #[test]
    fn test_len_counts_alive_not_slots() {
        let mut m = AliveMap::new();
        m.insert(1, "a");
        m.insert(2, "b");
        m.insert(3, "c");
        assert_eq!(m.entries.len(), 3);
        assert_eq!(m.len(), 3);
        m.remove(&2);
        assert_eq!(m.entries.len(), 3); // slot still there
        assert_eq!(m.len(), 2); // alive_count decreased
    }

    #[test]
    fn test_shrink_to_fit_alive_compacts_and_preserves_order() {
        let mut m = AliveMap::with_capacity(10);
        for i in 0..8 {
            m.insert(i, i * 10);
        }
        // remove some
        assert_eq!(m.remove(&2), Some(20));
        assert_eq!(m.remove(&5), Some(50));
        let before_slots = m.entries.len();
        assert!(before_slots == 8);
        assert_eq!(m.len(), 6);

        m.shrink_to_fit_alive();
        // now entries.len() should equal alive_count
        assert_eq!(m.entries.len(), m.len());
        // order of survivors should be preserved (original insertion order among alive)
        let survivors: Vec<_> = m.iter().map(|(k, v)| (*k, *v)).collect();
        let expected: Vec<_> = vec![(0, 0), (1, 10), (3, 30), (4, 40), (6, 60), (7, 70)];
        assert_eq!(survivors, expected);
    }

    #[test]
    fn test_iter_exact_size_and_fused() {
        let mut m = AliveMap::new();
        m.insert(1, "a");
        m.insert(2, "b");
        m.insert(3, "c");
        m.remove(&2);
        let mut it = m.iter();
        assert_eq!(it.len(), 2);
        assert_eq!(it.next(), Some((&1, &"a")));
        assert_eq!(it.len(), 1);
        assert_eq!(it.next(), Some((&3, &"c")));
        assert_eq!(it.len(), 0);
        assert_eq!(it.next(), None);
        // fused: subsequent next() continue returning None
        assert_eq!(it.next(), None);
    }
}
