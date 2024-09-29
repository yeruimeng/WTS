import os
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_path = 'gpt2'
model = GPT2ForSequenceClassification.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


model.to(device)
model.eval()


test_df = pd.read_csv('./data')


def preprocess(prompt):
    context = "Given a prompt, determine if it is safe or unsafe. "
    input_text = context + "Prompt: " + prompt
    inputs = tokenizer(input_text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    return {k: v.to(device) for k, v in inputs.items()}


predictions = []
with torch.no_grad():
    for prompt in test_df['prompt']:
        inputs = preprocess(prompt)
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        predictions.append(pred)


predictions_bool = [bool(pred) for pred in predictions]


accuracy = accuracy_score(test_df['is_safe'], predictions_bool)
print(f"Accuracy: {accuracy:.4f}")


output_df = pd.DataFrame({
    'prompt': test_df['prompt'],
    'is_safe': predictions_bool
})

output_df.to_csv('./data', index=False)
print("Predictions saved to the file")