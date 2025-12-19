
import json
import os

def main():
    config_path = "local_ckpt/hf_model_iter_0001000/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Old Config: hidden_size={config.get('hidden_size')}, intermediate_size={config.get('intermediate_size')}")

    # Deductions from checkpoint shapes:
    # mlp.gate_proj: [4864, 896] -> intermediate_size=4864, hidden_size=896
    # self_attn.q_proj: [896, 896] -> hidden_size=896
    # self_attn.k_proj: [128, 896] -> kv_dim=128
    
    config["hidden_size"] = 896
    config["intermediate_size"] = 4864
    
    # head_dim is 128 in original config.
    # If we keep head_dim=128:
    # num_attention_heads = 896 / 128 = 7
    # num_key_value_heads = 128 / 128 = 1
    
    # Verify if these make sense for Qwen models.
    # Qwen 0.5B: hidden=1024, heads=16 -> head_dim=64?
    # Qwen 1.5-0.5B: hidden=1024.
    # Qwen 2.5-0.5B: hidden=896? Let's assume the math holds.
    
    # Wait, assuming head_dim=128 might be wrong if the original config for 4B had it.
    # But K proj is 128. If num_key_value_heads > 1, then head_dim < 128.
    # If nu_key_value_heads=2, head_dim=64.
    # Then num_attention_heads = 896/64 = 14.
    
    # Qwen2.5-0.5B Config on HF:
    # hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2, head_dim=64.
    # This matches exactly! 14 * 64 = 896. 2 * 64 = 128.
    
    config["num_attention_heads"] = 14
    config["num_key_value_heads"] = 2
    # config["head_dim"] = 64 # Some configs don't explicitly state head_dim, implied by hidden/heads? 
    # Current config has "head_dim": 128. We should update it if it exists.
    if "head_dim" in config:
        config["head_dim"] = 64
        
    print(f"New Config: hidden_size={config['hidden_size']}, intermediate_size={config['intermediate_size']}, heads={config['num_attention_heads']}, kv_heads={config['num_key_value_heads']}")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Config patched.")

if __name__ == "__main__":
    main()
