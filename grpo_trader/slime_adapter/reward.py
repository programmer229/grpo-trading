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
    # 1. Parse the action from the response
    # Expected format: <think>...</think><answer>BUY</answer>
    response = sample.response
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    
    action_map = {"BUY": 1, "SELL": -1, "HOLD": 0}
    direction = 0
    
    # Format Reward: Small bonus for getting the syntax right
    reward = 0.1
    
    if match:
        action_str = match.group(1).strip().upper()
        
        # Log action distribution
        try:
            import wandb
            if wandb.run:
                metrics = {
                    "action/buy": 1.0 if "BUY" in action_str else 0.0,
                    "action/sell": 1.0 if "SELL" in action_str else 0.0,
                    "action/hold": 1.0 if "HOLD" in action_str else 0.0,
                }
                wandb.log(metrics)
        except ImportError:
            pass

        for act, val in action_map.items():
            if act in action_str:
                direction = val
                break
    else:
        # Penalty for invalid format (override positive reward)
        return -0.5
        
    # 2. Calculate reward based on price movement
    # Metadata should contain 'current_price' and 'next_price'
    metadata = sample.metadata
    if not metadata:
        return reward # Just format reward
        
    current_price = metadata.get('current_price')
    next_price = metadata.get('next_price')
    
    if current_price is None or next_price is None:
        return reward
        
    price_change_pct = (next_price - current_price) / current_price
    
    # PnL Reward = Direction * % Change
    # e.g. Buy: (+1 * change)
    # e.g. Sell: (-1 * change)
    # e.g. Hold: (0 * change) = 0
    
    pnl_component = (direction * price_change_pct)
    
    # Scale up to make it comparable to format reward
    reward += pnl_component * 100
    
    return float(reward)
