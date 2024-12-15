from transformers import LlamaForCausalLM, LlamaTokenizer
from milvinius import MilviniusClient
import numpy as np

# Step 1: Load Llama Model for Embedding
def load_llama_model(model_name="meta-llama/Llama-3-7b-hf"):
    print("Loading Llama model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

# Step 2: Embed Task Descriptions
def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs, output_hidden_states=True)
    # Extract the CLS token embedding
    return outputs.hidden_states[-1][:, 0, :].detach().numpy().squeeze()

# Step 3: Parse Input Data
def parse_input_data(file_path):
    tasks = []
    with open(file_path, "r") as file:
        for line in file:
            if ": Python Code:" in line:
                task, code = line.split(": Python Code:", 1)
                tasks.append({"task": task.strip(), "code": code.strip()})
    return tasks

# Step 4: Connect to Milvinius
def initialize_milvinius(endpoint="http://localhost:8080"):
    print("Connecting to Milvinius server...")
    client = MilviniusClient(endpoint)
    return client

# Step 5: Index Tasks and Code in Milvinius
def index_task_data(client, tasks, model, tokenizer):
    print("Indexing tasks into Milvinius...")
    for i, task_entry in enumerate(tasks):
        print(f"Processing task {i+1}/{len(tasks)}: {task_entry['task'][:50]}...")
        embedding = embed_text(task_entry["task"], model, tokenizer)
        client.index_vector(
            vector=embedding.tolist(),
            metadata={"id": i, "task": task_entry["task"], "code": task_entry["code"]}
        )
    print("All tasks indexed successfully.")

# Step 6: Main Function
if __name__ == "__main__":
    # Step 1: Load Llama model
    model, tokenizer = load_llama_model()

    # Step 2: Load and parse input data
    input_file_path = "tasks.txt"  # Path to your input file
    tasks = parse_input_data(input_file_path)

    # Step 3: Connect to Milvinius
    milvinius_client = initialize_milvinius()

    # Step 4: Index the tasks and corresponding code
    index_task_data(milvinius_client, tasks, model, tokenizer)
