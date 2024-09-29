import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class PromptContinuationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                prompt = item['prompt']['text']
                continuation = item['continuation']['text']
                self.data.append((prompt, continuation))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, continuation = self.data[idx]
        full_text = f"{prompt}{continuation}"
        encoding = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs):
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_epoch_{epoch + 1}.pt")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model_name = "openai-gpt"
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
    

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    

    model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    

    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Padding token: {tokenizer.pad_token}")
    print(f"Padding token ID: {tokenizer.pad_token_id}")

    
    full_dataset = PromptContinuationDataset("./data", tokenizer)
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.1, random_state=42)
    
    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    num_epochs = 3
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)  
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs=num_epochs)

    
    model.save_pretrained("./output")
    tokenizer.save_pretrained("./output")

if __name__ == "__main__":
    main()