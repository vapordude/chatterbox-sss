## 2024-10-08 - Fix Insecure Deserialization in PyTorch Model Loading
**Vulnerability:** `torch.load` used without `weights_only=True` when loading conditional states.
**Learning:** By default, PyTorch uses `pickle` for deserialization in `torch.load`, which allows execution of arbitrary code if loading a malicious model. This is particularly dangerous for models loaded from remote paths (like huggingface hub) or user uploads.
**Prevention:** Always use `weights_only=True` in `torch.load` to enforce loading only tensors, dictionaries and safe types.
