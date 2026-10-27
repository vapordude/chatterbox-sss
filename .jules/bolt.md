## 2024-05-24 - Hoist Static Tensor in Autoregressive Loops
**Learning:** In PyTorch autoregressive generation loops, creating a tensor (e.g. `torch.as_tensor()`) unconditionally inside the loop leads to redundant memory allocations and CPU-to-device transfers on every step, degrading performance.
**Action:** Use lazy initialization (`tensor = None` outside, initialize on first iteration) to hoist static tensor creations out of the generation loop while avoiding device/dtype mismatches.
