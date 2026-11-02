## YYYY-MM-DD - [Optimize tensor concatenation in T3]
**Learning:** PyTorch `torch.cat` with `dim=1` is significantly faster than doing list comprehension with `zip`, `cat`, and `stack` for batched tensor concatenations.
**Action:** Use `torch.cat([tensor1, tensor2, ...], dim=1)` instead of slow loop-based stacking when batch dimensions match.
