## 2023-10-27 - Insecure Deserialization via torch.load
**Vulnerability:** Found insecure deserialization via `torch.load` missing `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** While other components (like TTS models) safely loaded tensors, the Voice Conversion model missed this parameter. Since `conds.pt` contains dictionaries and tensors rather than complex objects, failing to enforce `weights_only=True` introduced an unnecessary critical risk if a user loaded a modified model file.
**Prevention:** Always default to `torch.load(..., weights_only=True)` unless full object deserialization is strictly required and the source is perfectly trusted.
