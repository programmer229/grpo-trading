import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from ..train.loss import compute_grpo_loss

class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        env,
        optimizer,
        group_size=4,
        beta=0.01,
        clip_eps=0.2,
        device="cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.optimizer = optimizer
        self.group_size = group_size
        self.beta = beta
        self.clip_eps = clip_eps
        self.device = device
        
        # Create reference model (frozen copy)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
    def train_step(self, batch_data):
        """
        Performs a single training step using GRPO.
        """
        prompts = batch_data['prompt'] # List of strings
        current_prices = batch_data['current_price']
        next_prices = batch_data['next_price']
        
        # 1. Sampling: Generate G completions for each prompt
        # We process one prompt at a time for simplicity in this demo, 
        # but in production you'd batch this.
        
        total_loss = 0
        
        for i, prompt in enumerate(prompts):
            # Duplicate prompt G times
            batch_prompts = [prompt] * self.group_size
            
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            # Generate completions
            # Note: We need to handle the fact that we want to train on the completion,
            # so we should generate and then concatenate, or use the model's generate method
            # and get the full sequence.
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode to get text for reward calculation
            completions_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # 2. Reward Calculation
            rewards = self.env.calculate_reward(
                completions_text, 
                current_prices[i], 
                next_prices[i]
            )
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            
            # 3. Advantage Calculation (Group Normalization)
            mean_reward = rewards_tensor.mean()
            std_reward = rewards_tensor.std() + 1e-8
            advantages = (rewards_tensor - mean_reward) / std_reward
            
            # 4. Compute Old Log Probs (of the generated sequence)
            # We treat the generated sequence as the "experience"
            # For the first step, old_policy == current_policy
            # In true PPO, we'd collect experience first, then train for K epochs.
            # GRPO often does this online or with a small buffer. 
            # Here we do a single update step per generation (online).
            
            # We need to compute log probs of the *generated* tokens under the *current* model (which is "old" for the backward pass)
            # Ideally we would have saved log_probs during generation, but HF generate doesn't always return them easily structured.
            # So we do a forward pass.
            
            with torch.no_grad():
                # We use the outputs (full sequence) as input
                # Create attention mask
                gen_attention_mask = (outputs != self.tokenizer.pad_token_id).long()
                
                fwd_outputs = self.model(input_ids=outputs, attention_mask=gen_attention_mask)
                logits = fwd_outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Gather log probs of the tokens
                token_log_probs = torch.gather(log_probs, 2, outputs.unsqueeze(-1)).squeeze(-1)
                if gen_attention_mask is not None:
                    token_log_probs = token_log_probs * gen_attention_mask
                old_sequence_log_probs = token_log_probs.sum(dim=1)
            
            # 5. Loss & Update
            # Now we compute gradients. We need to run the forward pass AGAIN with gradients enabled
            # on the SAME outputs.
            
            # Prepare inputs for loss function
            train_inputs = {
                'input_ids': outputs,
                'attention_mask': gen_attention_mask
            }
            
            loss, policy_loss, kl_loss = compute_grpo_loss(
                self.model,
                train_inputs,
                old_sequence_log_probs,
                advantages,
                ref_model=self.ref_model,
                beta=self.beta,
                clip_eps=self.clip_eps
            )
            
            # Accumulate gradients
            loss.backward()
            total_loss += loss.item()
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss / len(prompts)
