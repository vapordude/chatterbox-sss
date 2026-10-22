## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.

## 2024-05-24 - Utilizing Gradio 'info' parameter for helper text
**Learning:** Bloating component labels with instructional text (e.g. "max chars 300", "if none...") causes cluttered UI, poor screen reader experiences, and confusing visual hierarchy.
**Action:** Always use Gradio's native `info` parameter for explanatory helper text instead of cramming it into the `label`. Rely on built-in features like `max_length` to provide clean inline context.
