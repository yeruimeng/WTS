import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from transformers import EarlyStoppingCallback
import numpy as np
import random

# 限定只使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 确定是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


df = pd.read_csv('./data')


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')
special_tokens_dict = {'pad_token': '[PAD]', 'sep_token': '[SEP]'}
tokenizer.add_special_tokens(special_tokens_dict)


model = GPT2ForSequenceClassification.from_pretrained('openai-community/gpt2', num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)  

def preprocess_function(examples):
    inputs = examples['prompt']
    encodings = tokenizer(inputs, padding='max_length', max_length=512, truncation=True)
    labels = [1 if a == True else 0 for a in examples['is_safe']]
    encodings['labels'] = labels
    return {k: torch.tensor(v).to(device) for k, v in encodings.items()}

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

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

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,
)


optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)


total_steps = len(tokenized_train) * training_args.num_train_epochs // training_args.per_device_train_batch_size
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),  
    callbacks=[early_stopping_callback] 
)


trainer.train()


trainer.save_model(output_dir='./output')
tokenizer.save_pretrained('./output')


test_results = trainer.evaluate(tokenized_test)
print("Test results:", test_results)