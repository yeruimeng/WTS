import json
import torch
from torch.utils.data import Dataset
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from tqdm import tqdm
import os
import random
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_test_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def generate_continuation(model, tokenizer, prompt, max_length=100, temperature=1.0, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            temperature=temperature
        )
    
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    
    continuation = generated_text[len(prompt):].strip()
    return continuation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running on CPU.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    
    model_path = "./model"
    tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)
    model = OpenAIGPTLMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    
    test_data = load_test_data("./data")

    
    for i in range(25):
        set_seed(i)  
        results = []
        for item in tqdm(test_data, desc=f"Generating continuations (Run {i:02d}/24)"):
            prompt = item['prompt']['text']
            continuation = generate_continuation(model, tokenizer, prompt, temperature=0.7, top_p=0.95)
            results.append({
                'prompt': {'text': prompt},
                'continuation': {'text': continuation}
            })

        
        output_file = "./output"
        with open(output_file, 'w') as f:
            for item in results:
                json.dump(item, f)
                f.write('\n')
        print(f"Generated continuations for run {i:02d} saved to {output_file}")

if __name__ == "__main__":
    main()