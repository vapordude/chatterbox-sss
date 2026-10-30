## 2024-05-24 - Insecure PyTorch Deserialization in VC Module
**Vulnerability:** Found `torch.load` used without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** Using `torch.load` without `weights_only=True` allows arbitrary code execution via Python's `pickle` module when loading untrusted model weights.
**Prevention:** Always use `weights_only=True` when calling `torch.load` or prefer safer serialization formats like `safetensors`.
