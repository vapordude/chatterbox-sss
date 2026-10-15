## 2024-11-06 - Enforce character limits natively in Gradio Textbox
**Learning:** When character limits are only specified in component labels (e.g., "(max chars 300)"), users do not receive immediate frontend validation or visual feedback. Using the native `max_length` parameter in `gr.Textbox` enables built-in character counts and hard limits, enhancing UX and preventing backend truncation surprises.
**Action:** Always use native parameters like `max_length` in UI frameworks to enforce limits, providing immediate visual feedback rather than relying solely on descriptive labels.
