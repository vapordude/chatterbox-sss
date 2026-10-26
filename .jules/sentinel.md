## 2024-10-24 - Insecure Deserialization in PyTorch
**Vulnerability:** Found `torch.load` without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** Default behavior of `torch.load` uses `pickle`, which can execute arbitrary code during deserialization of untrusted checkpoints.
**Prevention:** Always use `weights_only=True` parameter when loading PyTorch checkpoints with `torch.load` to restrict unpickling to tensors and safe types.
