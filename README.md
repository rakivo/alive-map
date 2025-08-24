# `alive-map`

[![Crates.io](https://img.shields.io/crates/v/alive-map.svg)](https://crates.io/crates/alive-map)
[![Docs.rs](https://docs.rs/alive-map/badge.svg)](https://docs.rs/alive-map)
[![License](https://img.shields.io/crates/l/alive-map.svg)](https://crates.io/crates/alive-map)

# **WARNING**: Early stage development. APIs are unstable and most likely to change.

An insertion-order-preserving hash map with O(1) `.remove`

`AliveMap<K, V, S>` is a hash map that keeps track of which entries are alive
and preserves insertion order by book-keeping it in a doubly-linked list.

Unlike a standard `HashMap`, removing an entry marks it as dead rather than removing it immediately, allowing resurrection and compacting later.

# Memory vs Speed Tradeoff

`AliveMap` trades RAM for speed: removed entries remain in memory until
`shrink_to_fit_alive` is called. This avoids frequent reallocations and
preserves insertion order efficiently, but increases memory usage if many
entries are removed and not compacted. So it's a win-win situation because
who cares about RAM in 2025?

## Complexities:
- O(1) Insert average
- O(1) Remove average
- O(1) Get/Contains average
- O(`alive_count`) Iteration
- O(`alive_count`) Compaction

# Memory Overhead: AliveMap vs std HashMap (64-bit)

## Per-entry overhead:

- std HashMap: ~24 B per entry (bucket pointer, hash, metadata)
- AliveMap: ~32 B per entry
  - key: sizeof(key)
  - value: sizeof(value)
  - prev pointer: 8 B
  - next pointer: 8 B
  - `.aliveness` tag: ~0-1 bit
  - `.index_map` entry (key -> usize): ~sizeof(K) + 8 B

Additional overhead:

- `.entries` Vec metadata: 24 B (pointer + length + capacity)
- `.index_map`: HashMap overhead (buckets, hashes, metadata)

# Examples

```rust
use alive_map::AliveMap;

let mut map = AliveMap::new();

// Insert entries
map.insert(1, "a");
map.insert(2, "b");
map.insert(3, "c");
assert_eq!(map.len(), 3); // three alive entries

// Remove an entry
assert_eq!(map.remove(&2), Some("b")); // remove key 2
assert_eq!(map.get(&2), None);         // key 2 is dead
assert_eq!(map.len(), 2);              // only two alive entries remain

// Iterate over alive entries in insertion order
let items: Vec<_> = map.iter().collect();
assert_eq!(items, vec![(&1, &"a"), (&3, &"c")]);

// Resurrect a dead entry by reinserting
map.insert(2, "B");       // key 2 resurrected with new value
assert_eq!(map.len(), 3); // three alive entries again

// Iteration still preserves insertion order for alive entries
let items: Vec<_> = map.iter().collect();
assert_eq!(items, vec![(&1, &"a"), (&3, &"c"), (&2, &"B")]);

// Compact internal storage, removing dead slots
map.shrink_to_fit_alive();
assert_eq!(map.len(), 3); // alive count unchanged
```
