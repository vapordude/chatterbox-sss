## 2024-05-24 - Enforcing Gradio Character Limits Natively
**Learning:** Relying solely on component labels (e.g., "max chars 300") for character limits fails to prevent users from over-typing and lacks immediate visual feedback.
**Action:** Always use Gradio's native `max_length` parameter on `gr.Textbox` components to provide real-time character counts and strict frontend validation, improving usability and reducing backend errors.
## 2024-10-25 - Separating Helper Text from Component Labels
**Learning:** Cluttering primary component labels (e.g., `gr.Slider(label="min_p || Newer Sampler...")`) with instructional text creates visually messy UIs and degrades accessibility for screen readers.
**Action:** Always use the native `info` parameter in Gradio components for explanatory text. This keeps labels concise (e.g., "Min P") and provides cleaner, more accessible helper text below the component.
