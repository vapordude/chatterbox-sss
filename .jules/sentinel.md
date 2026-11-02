## 2024-05-24 - Arbitrary Code Execution via torch.load
**Vulnerability:** Found `torch.load` without `weights_only=True` in `src/chatterbox/vc.py`. This allows arbitrary code execution via insecure deserialization of Python pickle objects.
**Learning:** `torch.load` defaults to `weights_only=False` in older PyTorch versions, requiring explicit declaration of `weights_only=True` to prevent arbitrary code execution when loading untrusted model weights. Other files had this fixed but it was missed in `vc.py`.
**Prevention:** Always explicitly set `weights_only=True` when using `torch.load` or prefer safer alternatives like `safetensors`.
