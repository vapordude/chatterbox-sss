## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-05-25 - Decluttering Gradio Component Labels
**Learning:** Cluttering primary component `label` attributes with instructional text (e.g., "min_p || Newer Sampler...") creates visual noise and degrades screen reader accessibility.
**Action:** Always use the native `info` parameter in Gradio components for instructional or explanatory helper text to maintain a clean UI and ensure proper semantic separation of labels and descriptions.
