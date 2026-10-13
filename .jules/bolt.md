## 2026-10-13 - Hoisting Static Tensors in Autoregressive Loops
**Learning:** In PyTorch autoregressive loops, creating static tensors (like `torch.as_tensor(cfg_weight)`) inside the generation loop causes redundant memory allocations and CPU-to-device transfers on every step.
**Action:** Always hoist static tensor creations outside of generation loops to prevent unnecessary overhead and host-device synchronizations.
