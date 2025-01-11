import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the LLaMA model and tokenizer
model_name = "facebook/llama-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_dog_response(user_input, dog_name):
    # Customize the prompt to generate responses in dog language
    prompt = f"The user asked: {user_input}\n{dog_name} the dog responds like this:"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert the response to dog language (e.g., "woof woof")
    dog_response = response.replace(" ", " woof ").strip() + "!"
    return dog_response

st.title("Dog Language Chatbot")
st.write("Chat with a bot that responds in dog language!")

dog_name = st.text_input("Enter your dog's name:")
user_input = st.text_input("Ask a question:")

if st.button("Send"):
    if dog_name and user_input:
        response = generate_dog_response(user_input, dog_name)
        st.write(f"{dog_name} says: {response}")
    else:
        st.write("Please enter both your dog's name and your question.")
