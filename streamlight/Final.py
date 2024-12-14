import streamlit as st
import pandas as pd
import openai

import os
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load a dataset
def load_dataset(file):
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Improved cleaning function to remove artifacts
def clean_generated_code(code):
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

# Generate Python code from the query
def generate_code(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python assistant. Respond only with valid Python code. Do not include explanations, comments, or triple quotes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        raw_code = response['choices'][0]['message']['content'].strip()
        return clean_generated_code(raw_code)  # Clean the code
    except Exception as e:
        st.error(f"Error generating code: {e}")
        return None

# Execute the Python code
def execute_python_code(code, data):
    try:
        local_context = {"data": data}
        exec(code, {}, local_context)
        return local_context.get("result", "No result variable defined.")
    except SyntaxError as e:
        st.error(f"Syntax error in generated code: {e}")
        st.code(code, language="python")  # Display problematic code
        return None
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return None

# Validate if the query refers to a valid column
def validate_query(data, query):
    # Normalize column names and query for case-insensitive matching
    columns = [col.lower() for col in data.columns]
    query_lower = query.lower()

    # Check if any column name is mentioned in the query
    for column in columns:
        if column in query_lower:
            return True

    # If no match found, return False
    return False



# Streamlit application
st.title("NLP-to-Python Data Analysis RADAR")
st.write("Upload a dataset, enter your query, and get the results!")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=["csv", "xlsx"])
if uploaded_file:
    data = load_dataset(uploaded_file)
    if data is not None:
        st.write("### Dataset Preview")
        st.write(data.head())

        # Query input
        query = st.text_area("Enter your query:")
        if st.button("Submit Query"):
            if not query:
                st.error("Please enter a query.")
            else:
                # Validate the query
                if not validate_query(data, query):
                    st.error(f"Your query does not refer to any valid column in the dataset. Available columns: {', '.join(data.columns)}")
                else:
                    # Proceed with generating and executing the code
                    prompt = f"Given a dataset as a Pandas DataFrame called 'data', {query}. Store the result in a variable called 'result'."
                    code = generate_code(prompt)
                    if code:
                        result = execute_python_code(code, data)
                        if result is not None:
                            st.write("### Result")
                            st.write(result)

