# utils.py
import ast

def clean_generated_code(code):
    """
    Cleans Python code by removing unnecessary artifacts such as triple quotes
    and Markdown-style code blocks.
    """
    lines = code.splitlines()
    cleaned_lines = []
    inside_triple_quotes = False

    for line in lines:
        # Toggle triple-quote flag
        if line.strip().startswith(("'''", '"""')):
            inside_triple_quotes = not inside_triple_quotes
            continue  # Skip the line with triple quotes

        # Skip lines inside triple quotes
        if inside_triple_quotes:
            continue

        # Skip Markdown-style code blocks
        if line.strip().startswith("```"):
            continue

        # Add cleaned line
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
