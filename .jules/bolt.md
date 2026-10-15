## 2026-10-15 - Prevent multithreading contention in positional encodings
**Learning:** Unconditional RwLock::write() calls in frequently accessed layers (like dynamically scaled cached tensors for positional encodings) cause significant multithreading contention during inference.
**Action:** Implement a double-checked locking pattern (checking with a read lock first, then dropping and acquiring a write lock) to avoid unnecessary write locks and improve performance.
