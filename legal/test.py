from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import csv
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from datasets import load_dataset
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = './model'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2ForSequenceClassification.from_pretrained(model_path)
model.to(device)  

def predict_and_save(input_csv, output_csv):
    dataset = load_dataset('csv', data_files=input_csv)['train']
    
    predicted_answers = []
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'text', 'answer'])
        writer.writeheader()
        
        for item in dataset:
            question = item['question']
            text = item['text']
            

            context = "Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question."
            premise = "Classify if the clause is relevant to answering the question."
            input_text = context + premise + "Question: " + question + " [SEP] Clause: " + text
            
           
            inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                output = model(**inputs).logits
            predicted_label = output.argmax().item()
            predicted_answer = 'Relevant' if predicted_label == 1 else 'Irrelevant'
            
       
            writer.writerow({'question': question, 'text': text, 'answer': predicted_answer})
            predicted_answers.append(predicted_answer)
    
    return predicted_answers

input_csv = './test'
output_csv = './output'

predicted_answers = predict_and_save(input_csv, output_csv)


dataset = load_dataset('csv', data_files=input_csv)['train']
original_answers = dataset['answer']

accuracy = accuracy_score(original_answers, predicted_answers)
balanced_accuracy = balanced_accuracy_score(original_answers, predicted_answers)
f1 = f1_score(original_answers, predicted_answers, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
