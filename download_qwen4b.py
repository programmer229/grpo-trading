
from huggingface_hub import snapshot_download
import os
import shutil

def main():
    model_id = "Qwen/Qwen3-4B-Thinking-2507"
    local_dir = "base_qwen_4b"
    
    print(f"Downloading {model_id} assets...")
    # Only download config and tokenizer files
    allow_patterns = ["*.json", "*.txt", "*.tiktoken", "*.model", "*.py"]
    snapshot_download(repo_id=model_id, local_dir=local_dir, allow_patterns=allow_patterns)
    
    target_dir = "local_ckpt/hf_model_iter_0000200"
    os.makedirs(target_dir, exist_ok=True)
    
    # We will copy these to a base dir first for mock_common_pt
    print(f"Assets downloaded to {local_dir}")

if __name__ == "__main__":
    main()
