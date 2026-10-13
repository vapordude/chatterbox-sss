## 2026-10-13 - Enforce text limits via UI instead of labels
**Learning:** Hardcoding maximum length hints into `label` descriptions relies entirely on users reading and respecting them, which often fails.
**Action:** When building or modifying Gradio UIs (e.g., `gr.Textbox`), enforce character limits using native parameters like `max_length` rather than solely specifying limits in component labels. This provides immediate frontend validation and visual character counts, enhancing UX.
