import unittest
import torch
from grpo_trader.env.trading_env import TradingEnvironment
from grpo_trader.train.loss import compute_grpo_loss

class MockModel(torch.nn.Module):
    def __init__(self, vocab_size=100):
        super().__init__()
        self.linear = torch.nn.Linear(10, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        # Dummy output
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, 100)
        return type('obj', (object,), {'logits': logits})

class TestGRPO(unittest.TestCase):
    def test_reward_calculation(self):
        env = TradingEnvironment()
        completions = [
            "<think>...</think><answer>BUY</answer>",
            "<think>...</think><answer>SELL</answer>",
            "<think>...</think><answer>HOLD</answer>",
            "Garbage output"
        ]
        current_price = 100
        next_price = 110 # +10%
        
        rewards = env.calculate_reward(completions, current_price, next_price)
        
        # Buy: 1 * 0.1 * 100 = 10
        # Sell: -1 * 0.1 * 100 = -10
        # Hold: 0
        # Garbage: -0.05
        
        self.assertAlmostEqual(rewards[0], 10.0)
        self.assertAlmostEqual(rewards[1], -10.0)
        self.assertAlmostEqual(rewards[2], 0.0)
        self.assertAlmostEqual(rewards[3], -0.05)

    def test_grpo_loss(self):
        model = MockModel()
        input_ids = torch.randint(0, 100, (2, 10)) # Batch 2, Seq 10
        attention_mask = torch.ones_like(input_ids)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        old_log_probs = torch.randn(2)
        advantages = torch.randn(2)
        
        loss, policy_loss, kl_loss = compute_grpo_loss(
            model, inputs, old_log_probs, advantages
        )
        
        # Just check if it runs and returns scalars
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(torch.is_tensor(policy_loss))
        self.assertTrue(torch.is_tensor(kl_loss))

if __name__ == '__main__':
    unittest.main()
