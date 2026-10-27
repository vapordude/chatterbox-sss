## 2026-10-27 - [Insecure Deserialization in PyTorch Load]
**Vulnerability:** [Insecure deserialization via `torch.load` missing `weights_only=True` in `src/chatterbox/vc.py`]
**Learning:** [Loading PyTorch `.pt` files without restricting to weights only can lead to arbitrary code execution if the checkpoint file is compromised or untrusted.]
**Prevention:** [Always enforce `weights_only=True` when using `torch.load` unless executing arbitrary code from a trusted source is explicitly required.]
