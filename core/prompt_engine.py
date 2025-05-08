import openai
import json
import re

def extract_json_from_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    else:
        raise ValueError("No valid JSON found in LLM response.")

def prompt_to_json(user_prompt, schema, tile_state):
    system_msg = f"""
You are a dashboard assistant. Given a user's request and a dataset schema, output a valid JSON object to control the dashboard layout.

Your response should be one of two types:

1. To create a new chart:
{{
  "action": "add_chart",
  "chart_type": "bar",
  "x_axis": "species",
  "y_axis": "petal_length",
  "filter": null,
  "style": {{"color": "blue"}},
  "target_tile": "next_available"
}}

2. To update an existing chart:
{{
  "action": "update_chart",
  "chart_type": "bar",
  "x_axis": "species",
  "y_axis": "petal_length",
  "filter": null,
  "style": {{"color": "red"}},
  "target_tile": 0
}}

IMPORTANT RULES:
- Only return a JSON object—no explanations or comments.
- The keys must be lowercase.
- If no filter is needed, set "filter" to null.
- "target_tile" must be "next_available" or a number from 0 to 3.

Schema:
{schema}

Current tiles in use: {tile_state}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]
    )

    return extract_json_from_response(response["choices"][0]["message"]["content"])
