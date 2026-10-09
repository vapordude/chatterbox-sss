## 2024-05-24 - Enforce Character Limits Using Native Component Attributes
**Learning:** When building or modifying Gradio UIs (e.g., `gr.Textbox`), relying solely on component labels (e.g., "max chars 300") to indicate character limits leads to poor UX. Users lack immediate feedback when they exceed the limit until after submission or processing.
**Action:** Enforce character limits using native parameters like `max_length` (e.g., `max_length=300`) rather than solely specifying limits in component labels. This provides immediate frontend validation and visual character counts, enhancing UX.
