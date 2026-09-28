import os

py_files = []
for root, dirs, files in os.walk('.'):
    if '.git' in root or '.venv' in root or 'chatterbox-rs' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            # Ignore the generate_todo.py itself from the list of files to translate
            if file == 'generate_todo.py':
                continue
            py_files.append(os.path.normpath(os.path.join(root, file)))

py_files.sort()

with open('todo.md', 'w') as f:
    f.write("# Source of Truth for Rust Translation\n\n")
    f.write("Instructions for agents:\n")
    f.write("Translate the python files below to rust, **one python file at a time**. \n")
    f.write("Use **no external libraries**; use **first principle math first**.\n")
    f.write("Check against the python file for parity, then mark that file for the next agent to know by checking the box.\n\n")
    f.write("## Files:\n\n")
    for file in py_files:
        f.write(f"- [ ] `{file}`\n")

print("todo.md generated successfully.")
