import json
import torch
from torch.utils.data import Dataset
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from tqdm import tqdm
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_test_data(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def generate_continuation(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    
   
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=False,  
            temperature=0.0   
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

    results = []
    for item in tqdm(test_data, desc="Generating continuations"):
        prompt = item['prompt']['text']
        continuation = generate_continuation(model, tokenizer, prompt)
        results.append({
            'prompt': {'text': prompt},
            'continuation': {'text': continuation}
        })

    output_file = "./output"
    with open(output_file, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    print(f"Generated continuations saved to {output_file}")

if __name__ == "__main__":
    main()