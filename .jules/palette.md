
## 2023-10-27 - Gradio Textbox maxlength property
**Learning:** Enforcing character limits on `gr.Textbox` by relying solely on the input label text (e.g., "(max chars 300)") is poor UX. Using the native `max_length` parameter in Gradio 4+ adds an HTML `maxlength` attribute, providing immediate frontend validation and preventing users from overtyping, which is much better for usability and accessibility.
**Action:** When building or modifying Gradio UIs, always enforce character limits using native component parameters (like `max_length`) rather than just specifying them in textual labels.
