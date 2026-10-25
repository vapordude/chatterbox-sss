
## $(date +%Y-%m-%d) - PyTorch Autoregressive Tensor Creation Hoisting
**Learning:** In PyTorch token-by-token generation loops, static tensors derived from loop-invariant parameters (like `cfg_weight`) are often repeatedly initialized using `torch.as_tensor()` inside the hot loop. This causes redundant CPU-to-GPU syncs and memory allocations on every decoding step.
**Action:** When optimizing autoregressive loops, hoist static tensor creations outside the generation loop. If the target device or dtype is only determined by tensors produced within the loop, use lazy initialization (`var = None` outside, `if var is None: var = ...` inside) to avoid device mismatches while still achieving the optimization.
