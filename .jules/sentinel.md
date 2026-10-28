## 2024-10-28 - Insecure Deserialization in PyTorch Checkpoint Loading
**Vulnerability:** `torch.load` used without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** PyTorch's `torch.load` uses Python's `pickle` module by default, which allows arbitrary code execution if a malicious checkpoint is loaded.
**Prevention:** Always explicitly include the `weights_only=True` parameter when calling `torch.load` to restrict unpickling to safe types only.
