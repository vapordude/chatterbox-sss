## 2024-10-12 - Prevent Multithreading Contention in Cached Tensors
**Learning:** Using unconditional `RwLock::write()` for dynamically scaled cached tensors (like positional encodings in `chatterbox-rs/src/s3gen/transformer/embedding.rs`) causes severe multithreading contention during inference.
**Action:** Always implement a double-checked locking pattern (read lock first, check condition, drop read lock, then acquire write lock and double-check) for lazily initialized or dynamically scaled shared states in frequently accessed layers.
