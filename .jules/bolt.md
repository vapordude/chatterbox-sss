
## 2024-10-29 - Hoisting Static Tensors in PyTorch Loops
**Learning:** In PyTorch autoregressive loops, repeatedly creating static tensors (e.g., `torch.as_tensor(cfg_weight)`) on every step causes redundant memory allocations and CPU-to-device transfers, which creates a noticeable performance bottleneck.
**Action:** Hoist static tensor creations out of the generation loop. Use lazy initialization (e.g., `tensor = None` outside, initialize on the first iteration) when the exact device or dtype is only known from tensors produced inside the loop, rather than guessing from unrelated inputs.
