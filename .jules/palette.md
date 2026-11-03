## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-11-03 - Separate Instructions from Component Labels
**Learning:** Cluttering primary component labels with instructional text (e.g., "Exaggeration (Neutral = 0.5...)") increases cognitive load and results in poorer screen reader experiences, as the accessible name becomes bloated.
**Action:** When building or modifying Gradio UI components (e.g., sliders, numbers, checkboxes), use the native `info` parameter for instructional or explanatory helper text rather than cluttering the primary `label`, resulting in cleaner UI and better accessibility.
