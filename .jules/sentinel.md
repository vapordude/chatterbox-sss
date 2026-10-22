## 2024-05-24 - Insecure Deserialization in torch.load
**Vulnerability:** Found `torch.load` being used without `weights_only=True` in `src/chatterbox/vc.py`.
**Learning:** PyTorch's `torch.load` uses `pickle` by default, which allows arbitrary code execution if loading a malicious model file.
**Prevention:** Always use `weights_only=True` when calling `torch.load` to restrict unpickling to tensors and basic types only.
