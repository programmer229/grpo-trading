
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import sglang
from sglang import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--tp-size", type=int, default=1)
    return parser.parse_args()

def parse_action(response_text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        action_str = match.group(1).strip().upper()
        if "BUY" in action_str: return "BUY"
        if "SELL" in action_str: return "SELL"
    return "HOLD"

def simulate_trading(results, initial_capital=100000.0, risk_per_trade_pct=2):
    capital = initial_capital
    equity_curve = []
    trades = []
    
    for i, res in enumerate(results):
        prompt = res['prompt']
        response = res['response']
        metadata = res['metadata']  # Ensure your test data has metadata!
        
        action = parse_action(response)
        
        # metadata structure depends on how gen_data saved it
        # Assuming metadata has 'current_price' and 'next_price'
        current_price = metadata.get('current_price')
        next_price = metadata.get('next_price')
        
        if current_price is None or next_price is None:
            continue
            
        price_change_pct = (next_price - current_price) / current_price
        pnl = 0.0
        position_size = capital * risk_per_trade_pct
        
        if action == "BUY":
            pnl = position_size * price_change_pct
        elif action == "SELL":
            pnl = position_size * (-price_change_pct)
            
        capital += pnl
        
        equity_curve.append({
            'step': i, 'capital': capital, 'action': action,
            'price_change': price_change_pct, 'pnl': pnl
        })
        
        if action != "HOLD":
            trades.append({
                'step': i, 'action': action, 'pnl': pnl,
                'return_pct': (pnl / position_size) * 100 if position_size > 0 else 0
            })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    data = []
    with open(args.test_data, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples.")

    # Sort data by timestamp
    print("Sorting data by timestamp...")
    def get_timestamp(item):
        # Extract the last "Time: YYYY-MM-DD HH:MM:SS" from the prompt
        prompt_text = item['prompt']
        if isinstance(prompt_text, list): # Handle chat format if needed, though usually string here
            prompt_text = str(prompt_text)
            
        matches = re.findall(r"Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", prompt_text)
        if matches:
            return matches[-1] # Return the last timestamp found (latest in history)
        return "0000-00-00 00:00:00"

    data.sort(key=get_timestamp)
    print("Data sorted.")
    
    # Init SGLang
    print(f"Initializing SGLang Engine with {args.model_path}...")
    engine = Engine(model_path=args.model_path, tp_size=args.tp_size, trust_remote_code=True)
    
    # Load Tokenizer for Chat Template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Generate
    print("Generating responses...")
    prompts = []
    for i, d in enumerate(data):
        raw_prompt = d.get('prompt')
        # Check if it's already a string or needs template application
        if isinstance(raw_prompt, str):
             prompts.append(raw_prompt)
        elif isinstance(raw_prompt, list):
             # Assume chat messages
             try:
                 formatted_prompt = tokenizer.apply_chat_template(raw_prompt, tokenize=False, add_generation_prompt=True)
                 prompts.append(formatted_prompt)
             except Exception as e:
                 print(f"Error applying chat template for sample {i}: {e}")
                 # Fallback to string representation (will likely fail model generation but avoids crash)
                 prompts.append(str(raw_prompt))
        else:
             print(f"Warning: Unknown prompt format sample {i}: {type(raw_prompt)}")
             prompts.append(str(raw_prompt))

    if prompts:
        print(f"First prompt preview: {prompts[0][:100]}...")
    
    # sampling params
    sampling_params = {"temperature": 0.0, "max_new_tokens": 512}
    
    outputs = engine.generate(prompts, sampling_params)
    
    results = []
    for i, out in enumerate(outputs):
        results.append({
            'prompt': prompts[i],
            'response': out['text'],
            'metadata': data[i].get('metadata', {})
        })

    # Debug print first 5 outputs
    print("\n--- Model Output Debug (First 5) ---")
    for i in range(min(5, len(results))):
        print(f"Sample {i}:")
        print(f"Prompt: {results[i]['prompt'][:100]}...")
        print(f"Response: {results[i]['response']}")
        print("-" * 40)
    print("------------------------------------\n")
        
    # Save raw results
    with open(os.path.join(args.output_dir, "results.jsonl"), 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Simulate
    df_equity, df_trades = simulate_trading(results)

    # Print action counts
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for res in results:
        action = parse_action(res['response'])
        if action in action_counts:
            action_counts[action] += 1
        else:
            action_counts[action] = 1 # Should not happen with current logic but safe to have
            
    print("\n--- Action Distribution ---")
    print(f"BUY:  {action_counts.get('BUY', 0)}")
    print(f"SELL: {action_counts.get('SELL', 0)}")
    print(f"HOLD: {action_counts.get('HOLD', 0)}")
    print("---------------------------\n")
    
    if not df_equity.empty:
        final_capital = df_equity.iloc[-1]['capital']
        total_return = (final_capital - 100000) / 100000 * 100
        
        # Calculate Sharpe Ratio (assuming hourly data)
        df_equity['returns'] = df_equity['capital'].pct_change()
        # Annualized Sharpe: mean/std * sqrt(365*24)
        # We use a small epsilon to avoid division by zero if flat
        std_returns = df_equity['returns'].std()
        if std_returns > 1e-9:
            sharpe_ratio = (df_equity['returns'].mean() / std_returns) * (365 * 24) ** 0.5
        else:
            sharpe_ratio = 0.0
            
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Plot
        plt.figure(figsize=(10,6))
        plt.plot(df_equity['step'], df_equity['capital'])
        plt.title(f"Return: {total_return:.2f}% | Sharpe: {sharpe_ratio:.2f}")
        plt.savefig(os.path.join(args.output_dir, "equity_curve.png"))
        print(f"Saved plot to {args.output_dir}/equity_curve.png")
    else:
        print("No trades executed or no valid metadata.")

if __name__ == "__main__":
    main()
