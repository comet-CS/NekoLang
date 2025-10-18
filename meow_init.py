import os

print()
filename = input("Enter file name [main.neko]: ").strip()
if filename == "":
    filename = "main.neko"

if os.path.exists(filename):
    ow = input(f'File "{filename}" exists. Overwrite? (y/N): ').strip().lower()
    if ow != "y":
        print("Aborted.")
        exit()

print()
print("Choose template:")
print("  1) Terminal Program")
print("  2) UI Program")
choice = input("Enter 1 or 2: ").strip()

if choice == "2":
    content = """// NekoLang UI Starter
ui.window("My App", 400, 300, "#222");
ui.label("Welcome to NekoLang UI", "white");
ui.button("Click me", "white", "green");
ui.show();
"""
else:
    content = """// NekoLang Terminal Starter
print "Hello from Neko!";
"""

with open(filename, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Created {filename}")
