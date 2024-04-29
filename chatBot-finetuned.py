import torch
from transformers import BertForQuestionAnswering, BertTokenizer

# Load the fine-tuned model and tokenizer
model_path = "finetuned_model"
model = BertForQuestionAnswering.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define a function to answer questions
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Get the predicted answer span
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
    
    # Get the most likely answer span
    start_idx = torch.argmax(start_scores)
    end_idx = torch.argmax(end_scores) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx])
    
    return answer

# Define a function to interact with the chatbot
def chat():
    print("Welcome to the Chatbot! Ask me anything or type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        else:
            context = user_input
            question = input("What do you want to ask about? ")
            answer = answer_question(question, context)
            print("Chatbot:", answer)

# Start the chat
chat()
