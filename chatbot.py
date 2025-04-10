import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Function handles the generation of the response
# from the model based on the input prompt
def generate_responcse(
        model,
        tokenizer,
        prompt,
        max_length=50,
        temperature=1.0,
        top_p=0.9,
        top_k=50,

):
    # Turn the text prompt into tokens
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    
    # Turn off gradient calculations
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode the generated tokens back into text
    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # Clean up the output it was generating unnecessaty tokens
    stop_tokens = ['\n', 'User:', 'Chat Bot:']
    for token in stop_tokens:
        idx = generated_text.find(token)
        if idx != -1:
            generated_text = generated_text[:idx].strip()
            break
    return generated_text

#function to append the input/output to the terminal
# and format the conversation history
def prompt(conversation_history, user_input):
    # Format the conversation history and user input
    conversation_history.append(f"User: {user_input}")
    # limit conversatoin history to last 3 exchanges
    recent_history = conversation_history[-6:]

    prompt = "\n".join(recent_history) + "\nChat Bot:"

    return prompt

def main():

    model_name = "gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    # Set 
    max_length = 100
    temperature = 1.0
    top_p = 0.9
    top_k = 50


    print(f"Loaded model '{model_name}' Chatbot is ready!. Type 'exit' to quit.\n")
    print("To change settings, type commands like '/set max_length 120' or '/set temperature 0.7'\n")

    conversation_history = [] 
    # Initialize conversation history
    # Start the conversation loop
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break
        # Check for settings commands
        if user_input.startswith('/set'):
            parts = user_input.split()
            if len(parts) == 3:
                setting = parts[1]
                value = parts[2]

                if setting == 'max_length':
                    try:
                        max_length = int(value)
                        print(f"Max length set to {max_length}")
                    except ValueError:
                        print("Invalid value for max_length. Please enter an integer.")
                    continue
                elif setting == 'temperature':
                    try:
                        temperature = float(value)
                        print(f"Temperature set to {temperature}")
                    except ValueError:
                        print("Invalid value for temperature. Please enter a float.")
                    continue
                else:
                    print(f"Unknown setting: {setting}")
                    continue
            else:
                print("Invalid command format. Use '/set <setting> <value>'")
                continue
        # Initialize prompt
        prompt_text = prompt(conversation_history, user_input)

        bot_response = generate_responcse(
            model,
            tokenizer,
            prompt_text,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        conversation_history.append(f"Chat Bot: {bot_response}")
        print(f"Chat Bot: {bot_response}")

if __name__ == "__main__":
    main()
