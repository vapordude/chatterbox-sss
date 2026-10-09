## 2024-05-24 - Double-Checked Locking on Cached Tensors
**Learning:** Using an unconditional `RwLock::write()` call inside highly-frequent model execution loops (like `extend_pe` in positional encoding layers) causes significant multithreading contention during concurrent inference because the cache is already populated the vast majority of the time.
**Action:** Implement a double-checked locking pattern (read lock first, then conditionally acquire write lock on cache miss) for any dynamically scaled cached tensors in Rust to ensure lock contention is minimized.
