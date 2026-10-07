## 2025-02-28 - Enforce Character Limits Natively
**Learning:** In Gradio applications, appending length constraints like "(max chars 300)" to UI labels does not natively restrict user input length. This creates an accessibility and UX issue where users may overtype and only discover the limit upon submission.
**Action:** When building or modifying Gradio UIs (e.g., `gr.Textbox`), enforce character limits using native parameters like `max_length` rather than solely specifying limits in component labels. This provides immediate frontend validation and visual character counts, enhancing UX.
