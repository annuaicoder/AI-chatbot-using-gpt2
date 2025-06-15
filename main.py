from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("🤖 GPT-2 Chatbot is ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break

    # Encode the input and generate response
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

    # Generate response with sampling for diversity
    output_ids = model.generate(
        input_ids,
        max_length=150,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Display only the new part of the generated text
    bot_reply = response[len(user_input):].strip()
    print("Bot:", bot_reply)
