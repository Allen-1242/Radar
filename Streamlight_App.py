import streamlit as st
import pandas as pd
from milvinius import MilviniusClient
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils import clean_generated_code  # Import the cleaning function

# Initialize Milvinius and Llama
MILVINIUS_URL = "http://localhost:8080"  # Milvinius server endpoint
milvinius_client = MilviniusClient(MILVINIUS_URL)

# Load Llama model and tokenizer
def load_llama_model():
    model_name = "meta-llama/Llama-2-7b-hf"  # Adjust to your Llama model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

model, tokenizer = load_llama_model()

# Function to fetch data from Milvinius
def fetch_data_from_milvinius():
    try:
        st.info("Fetching data from Milvinius...")
        response = milvinius_client.query_vector(query_text="dataset", top_k=100)  # Adjust query and top_k as needed
        documents = [result['metadata']['text'] for result in response['results']]
        return pd.DataFrame({"Text": documents})  # Example DataFrame from results
    except Exception as e:
        st.error(f"Error fetching data from Milvinius: {e}")
        return None

# Generate Python code from the query using Llama
def generate_code_with_llama(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs["input_ids"], max_length=200, temperature=0)
    raw_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_generated_code(raw_code)  # Clean the code using external function

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

# Streamlit application
st.title("NLP-to-Python Data Analysis with Llama and Milvinius")
st.write("Fetch data from Milvinius, enter your query, and get results!")

# Fetch data from Milvinius
if st.button("Fetch Data from Milvinius"):
    data = fetch_data_from_milvinius()
    if data is not None:
        st.write("### Dataset Preview")
        st.write(data.head())

        # Query input
        query = st.text_area("Enter your query:")
        if st.button("Submit Query"):
            if not query:
                st.error("Please enter a query.")
            else:
                # Proceed with generating and executing the code
                prompt = f"Given a dataset as a Pandas DataFrame called 'data', {query}. Store the result in a variable called 'result'."
                code = generate_code_with_llama(prompt)
                if code:
                    result = execute_python_code(code, data)
                    if result is not None:
                        st.write("### Result")
                        st.write(result)
