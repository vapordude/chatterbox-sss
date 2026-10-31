## 2025-02-12 - [Insecure Deserialization in VC Module]
**Vulnerability:** Found `torch.load` being used without `weights_only=True` in `src/chatterbox/vc.py` when loading state checkpoints.
**Learning:** Using `torch.load` without this flag enables unsafe deserialization which can execute arbitrary code embedded in a malicious `.pt` file.
**Prevention:** Always add `weights_only=True` when using `torch.load()` or use `safetensors` instead when possible.
