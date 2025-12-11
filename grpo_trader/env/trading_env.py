import re

class TradingEnvironment:
    def __init__(self):
        self.actions = {"BUY": 1, "SELL": -1, "HOLD": 0}
        
    def parse_action(self, text):
        """
        Extracts the action from the <answer> tag.
        """
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
        if match:
            action_str = match.group(1).strip().upper()
            # Handle potential extra text or punctuation
            for act in self.actions:
                if act in action_str:
                    return self.actions[act]
        return None # Invalid format

    def calculate_reward(self, completions, current_price, next_price, **kwargs):
        """
        Calculates rewards for a batch of completions.
        
        Args:
            completions: List of generated strings.
            current_price: Float, price at time T.
            next_price: Float, price at time T+1.
            
        Returns:
            List of rewards (floats).
        """
        rewards = []
        price_change_pct = (next_price - current_price) / current_price
        
        for text in completions:
            direction = self.parse_action(text)
            
            if direction is None:
                # Penalty for invalid format
                rewards.append(-0.05) 
            else:
                # Reward is profit % * direction
                # Scale up a bit to make it significant for the loss
                reward = direction * price_change_pct * 100 
                rewards.append(reward)
                
        return rewards
