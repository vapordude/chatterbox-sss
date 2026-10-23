## 2024-05-14 - PyTorch Autoregressive Loop Micro-optimization
**Learning:** In PyTorch autoregressive generation loops, creating static tensors (like `torch.as_tensor(cfg_weight)`) inside the loop causes redundant memory allocations and CPU-to-device transfers on every token step, creating a measurable performance bottleneck.
**Action:** Always hoist static tensor creations outside of generation loops to prevent redundant allocations and host-device synchronizations.
