## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-10-27 - Using Info Parameter for UI Components
**Learning:** Cluttering the primary `label` of UI components (like Sliders and Numbers) with instructional or explanatory text degrades the user experience and accessibility.
**Action:** Use the native `info` parameter in Gradio components for instructional or explanatory helper text to provide a cleaner UI and better accessibility.
