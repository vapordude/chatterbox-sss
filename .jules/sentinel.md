## 2024-10-24 - [Fix torch.load insecure deserialization in vc.py]
**Vulnerability:** Found `torch.load` used without `weights_only=True` when loading `conds.pt` in `src/chatterbox/vc.py`, allowing insecure deserialization (RCE).
**Learning:** Even within the same project, some `torch.load` calls might be missing `weights_only=True` while others have it.
**Prevention:** Always use `weights_only=True` when loading PyTorch model checkpoints or states with `torch.load`, unless loading complex objects, in which case a safe alternative like Safetensors should be used.
