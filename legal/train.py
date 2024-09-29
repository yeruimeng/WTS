import os
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_path = "/train_set.csv"
dataset = load_dataset('csv', data_files=data_path)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens_dict = {'pad_token': '[PAD]', 'sep_token': '[SEP]'}
tokenizer.add_special_tokens(special_tokens_dict)


model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)  

def preprocess_function(examples):
    context = "Given a question and a clause from a privacy policy, determine if the clause contains enough information to answer the question."
    premise = "Classify if the clause is relevant to answering the question."
    inputs = [context + premise + "Question: " + q + " [SEP] Clause: " + t for q, t in zip(examples['question'], examples['text'])]
    encodings = tokenizer(inputs, padding='max_length', max_length=512, truncation=True)
    labels = [1 if a == 'Relevant' else 0 for a in examples['answer']]
    encodings['labels'] = labels
    

    return {k: torch.tensor(v).to(device) for k, v in encodings.items()}


tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,
    deepspeed=None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(output_dir='./output')
tokenizer.save_pretrained('./output')
