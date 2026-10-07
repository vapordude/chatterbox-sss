## 2023-10-27 - Double-Checked Locking for RwLock Cache
**Learning:** In Candle/Rust inference paths, using unconditional `RwLock::write()` on cached tensors (like Positional Encodings) creates massive multithreading contention because it blocks all readers during steady-state inference where the cache doesn't actually need expanding.
**Action:** Use double-checked locking: acquire `RwLock::read()` first, return early if the condition is met, and only acquire `RwLock::write()` if an update is genuinely necessary.
