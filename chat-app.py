from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.memory import ConversationBufferMemory
import torch

# Load Falcon-7B model
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize memory to store past conversations
memory = ConversationBufferMemory()

# Function to generate chatbot responses with memory
def generate_response(prompt):
    # Retrieve past messages from memory
    chat_history = memory.load_memory_variables({})["history"]

    # Combine chat history with the new prompt
    full_prompt = f"{chat_history}\nUser: {prompt}\nAssistant: "

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

    # Generate response
    output = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Save response to memory
    memory.save_context({"input": prompt}, {"output": response})

    return response

# Test chatbot
print(generate_response("Hello!"))
print(generate_response("Can you tell me what machine learning is?"))
print(generate_response("What did I ask before?"))
