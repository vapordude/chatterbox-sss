## 2024-10-04 - Enforcing Character Limits in Gradio UIs
**Learning:** Adding character limit instructions purely in the `label` of a `gr.Textbox` (e.g., "max chars 300") lacks built-in enforcement and user feedback. By utilizing the native `max_length` parameter on `gr.Textbox`, we instantly benefit from frontend input restriction and, crucially, a visual character counter that enhances UX and accessibility.
**Action:** When building or modifying Gradio UIs, always enforce character limits using native properties rather than relying solely on user-read instructions.
