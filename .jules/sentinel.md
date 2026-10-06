## 2024-10-06 - Insecure Deserialization in PyTorch Model Loading
**Vulnerability:** Found an instance of `torch.load` without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** In PyTorch, `torch.load` uses Python's `pickle` module by default. When loading untrusted files without `weights_only=True`, malicious payloads can execute arbitrary code. The memory mentioned this, but it's important to actively look for it.
**Prevention:** Always ensure `weights_only=True` is set when calling `torch.load`, especially when loading models or checkpoints from external sources.
