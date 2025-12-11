import torch
import torch.nn.functional as F

def compute_grpo_loss(
    model,
    inputs,
    old_log_probs,
    advantages,
    ref_model=None,
    beta=0.1,
    clip_eps=0.2
):
    """
    Computes the GRPO loss.
    
    Args:
        model: The policy model being trained.
        inputs: Tokenized inputs (input_ids, attention_mask) for the generated completions.
        old_log_probs: Log probabilities of the completions under the old policy (detached).
        advantages: Normalized advantages for each completion.
        ref_model: Reference model for KL penalty (optional).
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon.
        
    Returns:
        loss: Scalar tensor.
    """
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Forward pass to get current logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Calculate log probs of the generated tokens
    # We shift logits and labels by 1 because logits predict the *next* token
    # inputs['input_ids'] contains [Prompt, Completion]
    # We only care about the log probs of the Completion part, but for simplicity in this custom loop,
    # we might calculate it over the whole sequence and mask out the prompt later if needed.
    # For now, let's assume we calculate over the whole sequence but advantages are per-sequence.
    
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs of the actual tokens taken
    # input_ids: [Batch, SeqLen]
    # log_probs: [Batch, SeqLen, Vocab]
    
    # shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = input_ids[..., 1:].contiguous()
    
    # Use gather to get log_prob of the chosen token
    token_log_probs = torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probs over the sequence (or mean, depending on preference, usually sum for trajectory)
    # Masking padding tokens
    if attention_mask is not None:
        token_log_probs = token_log_probs * attention_mask
        
    # Sum over sequence length to get log_prob of the trajectory
    # shape: [Batch]
    sequence_log_probs = token_log_probs.sum(dim=1)
    
    # Ratio for PPO
    # old_log_probs should be shape [Batch]
    ratio = torch.exp(sequence_log_probs - old_log_probs)
    
    # PPO Clipped Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL Penalty
    kl_loss = 0
    if ref_model is not None:
        with torch.no_grad():
            ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            
        # KL(P || Ref) = sum(P * (logP - logRef))
        # Approximation: logP - logRef (since we are sampling from P)
        # This is the "approximated KL" often used in PPO
        # per token KL
        per_token_kl = torch.exp(log_probs) * (log_probs - ref_log_probs)
        if attention_mask is not None:
             per_token_kl = per_token_kl * attention_mask.unsqueeze(-1)
             
        kl_loss = per_token_kl.sum() / (attention_mask.sum() if attention_mask is not None else input_ids.numel())
    
    total_loss = policy_loss + beta * kl_loss
    
    return total_loss, policy_loss, kl_loss
