import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

# Load the fine-tuned model
model_path = "finetuned_model/BertFinetuned.model"
model = torch.load(model_path)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chatbot(query, context):
    # Tokenize the input
    inputs = tokenizer(context, query, return_tensors="pt", truncation=True, padding=True)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted start and end positions
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    
    # Print tokens and logits
    print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    print("Start Logits:", start_logits)
    print("End Logits:", end_logits)
    
    # Get the answer span
    answer_tokens = inputs["input_ids"][0][start_index : end_index + 1]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Example usage
context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."
query = "Who developed the theory of relativity?"
answer = chatbot(query, context)
print("Answer:", answer)




