import json
import sys
from transformers import AutoTokenizer

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = "train_data.jsonl"

def check_data_lengths():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Reading {DATA_FILE}...")
    with open(DATA_FILE, "r") as f:
        lines = f.readlines()

    print(f"Found {len(lines)} samples.")
    
    max_len = 0
    min_len = float('inf')
    skipped_512 = 0
    skipped_4096 = 0
    skipped_8192 = 0
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        prompt_data = data["prompt"]
        
        # Simulate Slime's _build_messages (simplified)
        if isinstance(prompt_data, str):
            messages = [{"role": "user", "content": prompt_data}]
        else:
            messages = prompt_data
            
        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(text)["input_ids"]
        length = len(tokens)
        
        max_len = max(max_len, length)
        min_len = min(min_len, length)
        
        if length > 512: skipped_512 += 1
        if length > 4096: skipped_4096 += 1
        if length > 8192: skipped_8192 += 1
        
        if i < 3:
            print(f"Sample {i} length: {length} tokens")
            
    print("-" * 30)
    print(f"Total Samples: {len(lines)}")
    print(f"Min Length: {min_len}")
    print(f"Max Length: {max_len}")
    print("-" * 30)
    print(f"Skipped if limit=512: {skipped_512} ({skipped_512/len(lines):.1%})")
    print(f"Skipped if limit=4096: {skipped_4096} ({skipped_4096/len(lines):.1%})")
    print(f"Skipped if limit=8192: {skipped_8192} ({skipped_8192/len(lines):.1%})")

if __name__ == "__main__":
    check_data_lengths()
