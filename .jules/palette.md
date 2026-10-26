## 2024-10-26 - Move instructional text to info parameter for Gradio UI components
**Learning:** Cluttering the primary `label` parameter with lengthy instructional text in Gradio makes the UI feel messy and is poor for accessibility. The native `info` parameter exists specifically to provide helper text without degrading the primary label's readability.
**Action:** Always use the `info` parameter for instructional or explanatory helper text in Gradio UI components (e.g., sliders, numbers, checkboxes) to maintain clean, accessible labels.
