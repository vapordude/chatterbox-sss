## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-10-30 - Cleaner UI and Better Accessibility with Gradio's `info` Parameter
**Learning:** Cluttering the `label` parameter of Gradio components (like sliders and numbers) with instructional helper text makes the UI messy and creates long, verbose labels for screen readers.
**Action:** Always use the native `info` parameter for instructional or explanatory helper text in Gradio UI components rather than stuffing it into the primary `label`.
