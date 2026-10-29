## 2024-05-24 - [Fix torch.load insecure deserialization]
**Vulnerability:** Found `torch.load` used without `weights_only=True` when loading `.pt` files.
**Learning:** The default behavior of `torch.load` allows arbitrary code execution via Python `pickle` if a malicious checkpoint file is loaded.
**Prevention:** Always use `weights_only=True` with `torch.load` when loading PyTorch checkpoints unless full pickling is explicitly needed and the source is 100% trusted.
