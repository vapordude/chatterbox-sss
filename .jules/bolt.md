## 2024-05-20 - Hoisting static tensor creation in autoregressive loops
**Learning:** In PyTorch autoregressive loops, operations like `torch.as_tensor` that create new tensors on every step cause redundant memory allocations and CPU-to-device synchronization, significantly slowing down token-by-token generation.
**Action:** Always hoist static tensor creations outside of generation loops. If the exact device or dtype is only known from tensors produced inside the loop, use lazy initialization (e.g., `tensor = None` outside, initialize on first iteration) to avoid mismatches.
