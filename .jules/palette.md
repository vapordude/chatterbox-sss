## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-11-01 - Use Gradio's info parameter instead of cluttered labels
**Learning:** Cluttering the `label` parameter with instructional or explanatory text in Gradio UI components leads to a messy UI and poor accessibility, as screen readers might read long strings unnecessarily or users might miss the primary input purpose.
**Action:** Use the native `info` parameter for instructional or explanatory helper text when building or modifying Gradio components (e.g., sliders, numbers, checkboxes) for a cleaner and more accessible interface.
