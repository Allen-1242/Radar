#We are testing this on the openAI API before we move over to the actual RAG
import ast

#Data and visualization validation libabries 

import pandas as pd
import numpy as npHello
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
def load_dataset(file_path):
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Clean generated code
def clean_generated_code(code):
    lines = code.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith(("python", "```"))]
    return "\n".join(cleaned_lines).strip()

# Generate Python code
def generate_code(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful Python assistant for data analysis. Only return the Python code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        raw_code = response['choices'][0]['message']['content'].strip()
        return clean_generated_code(raw_code)
    except Exception as e:
        print(f"Error generating code: {e}")
        return None

# Execute Python code
def execute_python_code(code, data):
    try:
        local_context = {"data": data}  # Provide dataset in execution context
        exec(code, {}, local_context)  # Execute code
        return local_context.get("result", None)  # Extract the "result" variable
    except Exception as e:
        print(f"Error executing code: {e}")
        return None

# Analyze dataset based on user query
def analyze_query(data, query):
    # Generate Python code for the query
    prompt = f"Given a dataset as a Pandas DataFrame called 'data', {query}. Store the result in a variable called 'result'."
    code = generate_code(prompt)
    if not code:
        return "Failed to generate Python code."

    print("\nGenerated Code:\n", code)  # Optional: Debugging

    # Execute the generated code
    result = execute_python_code(code, data)
    if result is None:
        return "Failed to execute the code."

    return result

# Main interactive loop
def interactive_analysis(file_path):
    # Load the dataset
    data = load_dataset(file_path)
    if data is None:
        print("Could not load dataset.")
        return

    print("Dataset loaded successfully! Start asking your questions about the data.\n")

    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        # Analyze the query
        result = analyze_query(data, query)
        print("\nResult:", result)



# Example Usage
# Assuming you have a dataset named "data.csv"
interactive_analysis(r"/Users/allensunny/myenv2/Irisdataset.csv")

