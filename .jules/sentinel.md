## 2024-05-18 - Insecure Deserialization in PyTorch Models
**Vulnerability:** The application used `torch.load()` without the `weights_only=True` parameter when loading PyTorch model checkpoints (`.pt` or `.pth` files) in `src/chatterbox/vc.py` and other modules.
**Learning:** PyTorch's `torch.load()` uses Python's `pickle` module by default, which is known to be insecure. Loading an untrusted checkpoint without `weights_only=True` can lead to arbitrary code execution if the file contains a malicious payload.
**Prevention:** Always include `weights_only=True` in `torch.load()` calls unless there is a specific, well-documented need to load arbitrary Python objects, in which case alternative, secure serialization formats (like `safetensors`) should be prioritized.
