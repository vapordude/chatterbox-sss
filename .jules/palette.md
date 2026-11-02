## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.
## 2024-11-02 - Clean Labels with Native Info Parameters
**Learning:** Cluttering primary Gradio component labels with instructional text (e.g., "min_p || Newer Sampler...") creates overly long descriptions for screen readers and visually cluttered UIs.
**Action:** Use Gradio's native `info` parameter for all explanatory or instructional helper text on UI components (sliders, numbers, textboxes), keeping the primary `label` concise and accessible.
