## 2023-10-25 - Enforce character limits with native parameter
**Learning:** When building Gradio UIs (e.g., `gr.Textbox`), relying solely on component labels (like "max chars 300") to enforce character limits is insufficient and lacks immediate visual feedback for the user.
**Action:** Enforce character limits using native parameters like `max_length`. This provides immediate frontend validation and visual character counts, enhancing UX by preventing users from exceeding the limit before submission.
