## 2024-10-22 - Tensor Allocation in Autoregressive Loops
**Learning:** In PyTorch autoregressive loops, placing static tensor creations (like `torch.as_tensor`) inside the loop causes redundant memory allocations and CPU-to-device transfers on every generation step, leading to significant overhead.
**Action:** Always hoist static tensor creations (e.g., CFG weight tensors) outside the token generation loop.
