## 2024-10-24 - [Insecure Deserialization in PyTorch]
**Vulnerability:** Found `torch.load()` without `weights_only=True` when loading local model weights in `src/chatterbox/vc.py`.
**Learning:** PyTorch uses Python's `pickle` for serialization, which allows arbitrary code execution (RCE) via insecure deserialization. Models downloaded from or verified by huggingface_hub might be safe, but loading unverified, user-provided local weights `torch.load("conds.pt")` exposes the app to critical RCE risks.
**Prevention:** Always include `weights_only=True` when calling `torch.load()` unless you explicitly need to load complex objects and absolutely trust the source file.
