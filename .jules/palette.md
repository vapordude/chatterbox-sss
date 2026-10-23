## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-05-24 - Gradio Component Helpers
**Learning:** Using the native `info` parameter for instructional text on Gradio components (instead of appending to `label`) provides a cleaner UI and improves screen-reader accessibility.
**Action:** Always prefer `info` over long `label` strings for secondary guidance on inputs.
