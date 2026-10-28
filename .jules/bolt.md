## 2024-05-24 - Double-checked locking for Positional Encodings
**Learning:** Unconditional `RwLock::write()` calls in frequently accessed layers (like dynamically scaled cached tensors for positional encodings) cause severe multithreading contention during inference.
**Action:** Always implement a double-checked locking pattern (check with a read lock first) before acquiring a write lock to avoid contention on shared model weights.
