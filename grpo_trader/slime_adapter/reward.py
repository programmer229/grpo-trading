import re
import torch

# Define the Sample class structure based on Slime's expectations (mock for type hinting if needed, 
# but in runtime Slime provides it)
# from slime.core.data import Sample 

async def reward_func(args, sample, **kwargs):
    """
    Custom reward function for GRPO Trading.
    
    Args:
        args: Training arguments.
        sample: The Sample object containing prompt, response, and metadata.
        
    Returns:
        float: The calculated reward.
    """
    # Debug logging
    print(f"[DEBUG] Reward Func Called")
    # print(f"[DEBUG] Prompt: {sample.prompt[:50]}...") 
    # print(f"[DEBUG] Response: {sample.response}")
    
    # 1. Parse the action from the response
    # Expected format: <think>...</think><answer>BUY</answer>
    response = sample.response
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    
    action_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    direction = 0
    
    if match:
        action_str = match.group(1).strip().upper()
        print(f"[DEBUG] Parsed Action: {action_str}")
        for act, val in action_map.items():
            if act in action_str:
                direction = val
                break
    else:
        print(f"[DEBUG] No match found in response")
        # Penalty for invalid format
        return -0.5
        
    # 2. Calculate reward based on price movement
    # Metadata should contain 'current_price' and 'next_price'
    metadata = sample.metadata
    if not metadata:
        print(f"[DEBUG] No metadata found")
        return 0.0
        
    current_price = metadata.get('current_price')
    next_price = metadata.get('next_price')
    
    if current_price is None or next_price is None:
        print(f"[DEBUG] Missing price data: current={current_price}, next={next_price}")
        return 0.0
        
    price_change_pct = (next_price - current_price) / current_price
    
    # Reward = Direction * % Change * Scale
    # e.g., Buy (1) * +0.01 (1%) * 100 = 1.0
    # e.g., Sell (-1) * +0.01 (1%) * 100 = -1.0
    reward = direction * price_change_pct * 100
    
    print(f"[DEBUG] Reward: {reward} (Dir: {direction}, Pct: {price_change_pct:.4f})")
    
    return float(reward)
