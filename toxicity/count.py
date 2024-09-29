def calculate_mean_toxicity(file_path):
    import json
    
    prompt_toxicities = []
    continuation_toxicities = []
    
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            
            prompt_toxicity = item['prompt'].get('toxicity')
            if prompt_toxicity and 'toxicity' in prompt_toxicity:
                prompt_toxicities.append(prompt_toxicity['toxicity'])
    
            continuation_toxicity = item['continuation'].get('toxicity')
            if continuation_toxicity and 'toxicity' in continuation_toxicity:
                continuation_toxicities.append(continuation_toxicity['toxicity'])
    
    mean_prompt_toxicity = sum(prompt_toxicities) / len(prompt_toxicities) if prompt_toxicities else None
    mean_continuation_toxicity = sum(continuation_toxicities) / len(continuation_toxicities) if continuation_toxicities else None
    
    return mean_prompt_toxicity, mean_continuation_toxicity


file_path = './output'
mean_prompt, mean_continuation = calculate_mean_toxicity(file_path)
print("Mean prompt toxicity:", mean_prompt)
print("Mean continuation toxicity:", mean_continuation)

