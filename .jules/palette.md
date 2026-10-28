## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-10-28 - Clean up UI with native info parameter
**Learning:** Cluttering the primary label of Gradio components (like sliders, textboxes) with instructional text reduces readability and accessibility.
**Action:** Use the native info parameter for instructional or explanatory helper text to create a cleaner UI and improve accessibility.
