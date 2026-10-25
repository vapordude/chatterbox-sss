## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-10-25 - Utilizing Info Parameter for Cleaner Gradio UIs
**Learning:** Embedding explanatory text directly within Gradio component `label` attributes clutters the UI and reduces visual hierarchy.
**Action:** Always use the native `info` parameter on Gradio components to house secondary instructional text, keeping labels concise and readable.
