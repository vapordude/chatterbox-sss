## 2023-10-31 - Hoist static tensor creations in autoregressive loops
**Learning:** In PyTorch autoregressive generation loops, creating tensors using `torch.as_tensor` inside the loop causes redundant memory allocations and CPU-to-device transfers on every step, degrading performance.
**Action:** Use lazy initialization (e.g., `tensor = None` outside, initialize on first iteration) to hoist static tensor creations outside of generation loops without risking device or dtype mismatches.
