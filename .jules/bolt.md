## 2024-10-16 - Hoist Tensor Initializations in Autoregressive Loops
**Learning:** In PyTorch autoregressive loops, static tensor creations (like `torch.as_tensor`) inside the generation loop cause redundant memory allocations and CPU-to-device transfers on every step, leading to performance degradation.
**Action:** Hoist static tensor creations outside the generation loop to prevent redundant operations and improve inference speed.
