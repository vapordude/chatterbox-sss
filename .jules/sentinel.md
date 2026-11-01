## 2024-11-01 - Insecure Deserialization in PyTorch Checkpoint Loading
**Vulnerability:** Found `torch.load()` without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** PyTorch's default `torch.load` uses pickle, which can execute arbitrary code upon deserialization. This was overlooked in this specific file while present in others.
**Prevention:** Always set `weights_only=True` when loading checkpoints with `torch.load()`.
