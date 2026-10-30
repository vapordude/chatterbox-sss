
## 2024-05-30 - Avoid tensor conversion in autoregressive loops
**Learning:** Converting Python integers to GPU tensors for positional embedding lookups in tight autoregressive loops causes severe performance regressions due to host-to-device synchronization.
**Action:** For integer index lookups in inference, use native integer scalar indexing on weights (e.g., `weight[idx:idx+1].unsqueeze(0)`) to avoid redundant tensor allocations and transfers.
