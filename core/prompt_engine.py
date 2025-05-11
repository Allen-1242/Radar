import openai
import json
import re
import streamlit as st

# Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extract_json_from_response(text: str) -> dict:
    """Pull the first JSON object out of a possibly-annotated LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No valid JSON found in LLM response.")
    return json.loads(match.group(0))


def prompt_to_json(user_prompt: str, schema: dict, tile_state: list) -> dict:
    """Send the user's prompt plus schema and tile-state to the LLM and return parsed JSON."""
    system_msg = f"""
You are a dashboard assistant. Given a user's request, a dataset schema, and current tile usage,
return a JSON with one of two actions:

1) To add a new chart:
{{
  "action": "add_chart",
  "chart_type": "bar|line|pie|scatter",
  "x_axis": "column_name",
  "y_axis": "column_name",       # omit for pie charts
  "filter": null|{{"col":value}},
  "style": {{ "color": "red" }}, # optional
  "title": "Chart Title",        # optional
  "target_tile": "next_available"  
}}

2) To update an existing chart:
{{
  "action": "update_chart",
  "chart_type": "bar|line|pie|scatter",
  "x_axis": "column_name",
  "y_axis": "column_name",       
  "filter": null|{{"col":value}},
  "style": {{ "color": "blue" }},# optional
  "title": "New Title",          # optional
  "target_tile": 0                # an integer 0-3
}}

Only return the JSON object—no explanation or extra text.

Dataset schema:
{schema}

Tiles in use (indices):
{tile_state}
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_prompt}
        ],
    )
    return extract_json_from_response(response.choices[0].message.content)


def inject_rag_context(user_prompt: str, rag_context: str) -> str:
    """Prepend a retrieval-augmented context block to the user prompt."""
    return f"""You are a data analysis assistant. Use the following data context to interpret the request.

DATA CONTEXT:
{rag_context}

USER PROMPT:
{user_prompt}"""
