import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_responcse(
        model,
        tokenizer,
        prompt,
        max_length=50,
        temperature=1.0,
        top_p=0.9,
        top_k=50,

):
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    

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

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return generated_text

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

    max_length = 100
    temperature = 1.0
    top_p = 0.9
    top_k = 50


    print(f"Loaded model '{model_name}' Chatbot is ready!. Type 'exit' to quit.\n")

    conversation_history = [] 

    while True:

        user_input = input("You: ")
        if user_input.lower().strip() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

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
