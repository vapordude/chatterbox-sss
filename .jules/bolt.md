## 2024-10-26 - [Hoisting static tensor creation in autoregressive loops]
**Learning:** In PyTorch autoregressive loops, static tensor creations (like `torch.as_tensor`) executed on every token generation step cause redundant CPU-to-device memory transfers and slow down inference. However, their exact dtype/device might only be known from dynamically produced tensors inside the loop.
**Action:** Use lazy initialization (`tensor = None` outside, initialize on first iteration) to safely hoist static tensor creation out of the loop without risking device/dtype mismatches.
