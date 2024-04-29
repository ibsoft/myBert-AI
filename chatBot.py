import json
import random
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering

# Define the bert tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Load the fine-tuned model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.eval()

# Save the model to disk
model.save_pretrained("bert_qa_model")

# Load the model from disk
loaded_model = BertForQuestionAnswering.from_pretrained("bert_qa_model")
loaded_model.eval()

# Load SQuAD 2.0 dataset
def load_squad_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    return squad_data['data']

# Extract contexts and questions from SQuAD 2.0 dataset
def extract_contexts_and_questions(dataset):
    contexts = []
    questions = []
    for topic in dataset:
        for passage in topic['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                contexts.append(context)
                questions.append(question)
    return contexts, questions

# Load SQuAD 2.0 dev dataset and extract contexts and questions
squad_dataset = load_squad_dataset("data/dev-v2.0.json")
contexts, questions = extract_contexts_and_questions(squad_dataset)

# Define a function to use the loaded model for question answering
def chatbot_qa(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
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
            # Randomly select a context from SQuAD dataset
            index = random.randint(0, len(contexts) - 1)
            context = contexts[index]
            question = user_input
            answer = chatbot_qa(question, context)
            print("Chatbot:", answer)

# Start the chat
chat()
