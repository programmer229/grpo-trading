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
    
    # Removed HOLD to force binary decision
    action_map = {"BUY": 1, "SELL": -1}
    direction = 0
    
    # Format Reward: Small bonus for getting the syntax right
    # Reduced to 0.01 to prioritize PnL
    reward = 0.01
    
    # Determine prefix based on data split (train/test)
    mode = metadata.get('split', 'train') 
    
    # Only log in Eval mode
    if mode == "test":
        try:
            import wandb
            if wandb.run:
                # 1. Action Distribution
                metrics = {
                    "eval/action/buy": 1.0 if "BUY" in action_str and match else 0.0,
                    "eval/action/sell": 1.0 if "SELL" in action_str and match else 0.0,
                }
                
                # 2. Model Reward Conditional on Action
                # If we successfully made a decision, track the reward for that specific decision
                if match:
                    if "BUY" in action_str:
                        metrics["eval/reward/model_buy"] = reward
                    elif "SELL" in action_str:
                        metrics["eval/reward/model_sell"] = reward

                # 3. Baseline Rewards (What if we always bought/sold?)
                # Baseline assumes correct format (+0.01)
                buy_reward = 0.01 + (1.0 * price_change_pct * 1000)
                sell_reward = 0.01 + (-1.0 * price_change_pct * 1000)
                
                metrics["eval/baseline/buy_reward"] = buy_reward
                metrics["eval/baseline/sell_reward"] = sell_reward
                
                wandb.log(metrics)
        except ImportError:
            pass
    
    return float(reward)
