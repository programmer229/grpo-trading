
import os
import ray
import re
import math
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from slime.ray.placement_group import create_placement_groups, create_rollout_manager
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
import sglang
from sglang.srt.server_args import ServerArgs
print(f"DEBUG: sglang version: {sglang.__version__}")
print(f"DEBUG: ServerArgs fields: {dir(ServerArgs)}")

def print_args(args):
    print("DEBUG: All args:")
    for arg in vars(args):
        if "sglang" in arg:
             print(f"  {arg}: {getattr(args, arg)}")



def parse_action(response_text):
    """Parses the BUY/SELL decision from the model response."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        action_str = match.group(1).strip().upper()
        if "BUY" in action_str:
            return "BUY"
        if "SELL" in action_str:
            return "SELL"
    return "HOLD"

def simulate_trading(samples, initial_capital=100000.0, risk_per_trade_pct=0.10):
    """
    Simulates trading based on model samples.
    
    Args:
        samples: List of Slime Sample objects (or dicts if converted).
        initial_capital: Starting cash (USD).
        risk_per_trade_pct: Percentage of current capital to allocate per trade.
        
    Returns:
        DataFrame with trade log and equity curve.
    """
    capital = initial_capital
    equity_curve = []
    trades = []
    
    # Sort samples if possible (though test set is usually ordered)
    # Assuming samples come in order or we trust the list order for now as simplistic replay
    
    for i, sample in enumerate(samples):
        # Handle both object and dict access
        if hasattr(sample, 'response'):
            response = sample.response
            metadata = sample.metadata
        else:
            response = sample['response']
            metadata = sample['metadata']
            
        action = parse_action(response)
        
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
            'step': i,
            'capital': capital,
            'action': action,
            'price_change': price_change_pct,
            'pnl': pnl
        })
        
        if action != "HOLD":
            trades.append({
                'step': i,
                'action': action,
                'pnl': pnl,
                'return_pct': (pnl / position_size) * 100 if position_size > 0 else 0
            })

    return pd.DataFrame(equity_curve), pd.DataFrame(trades)

def main(args):
    configure_logger()
    
    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # 2. Setup Slime Rollout
    # Create placement groups
    # args.num_gpus_per_node is used for rollout usually
    pgs = create_placement_groups(args)
    
    # Create rollout manager
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    
    # 3. Generate Rollouts
    print("Starting generation on test set...")
    # Rollout ID 0 is fine for one-off eval
    rollout_data_ref = ray.get(rollout_manager.generate.remote(0))
    
    # Get the actual data
    # rollout_data_ref is usually a list of ObjectRefs or the data itself depending on impl.
    # In Slime train.py: `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))`
    # If using dataset logic, it returns samples.
    
    # Note: ray.get() on the result of generate.remote() usually waits and returns the result.
    # train.py does: `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))`
    # Then it passes it to async_train.
    
    # Wait, `generate` returns a reference to data on the actors? Or the data itself?
    # Checking SGLang rollout source... it likely returns the samples.
    # Let's inspect type at runtime or assume it's the list of lists of samples.
    
    samples_flat = []
    # If it's a list of lists (batching), flatten it
    if isinstance(rollout_data_ref, list):
        for batch in rollout_data_ref:
            if isinstance(batch, list):
                samples_flat.extend(batch)
            else:
                samples_flat.append(batch)
    else:
        # Maybe it IS the data
        samples_flat = rollout_data_ref

    print(f"Generated {len(samples_flat)} samples.")
    
    # 4. Run Trading Simulation
    print("Running trading simulation...")
    df_equity, df_trades = simulate_trading(samples_flat)
    
    # 5. Analysis & Plotting
    final_capital = df_equity.iloc[-1]['capital']
    total_return = (final_capital - 100000) / 100000 * 100
    win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) if len(df_trades) > 0 else 0
    
    print("="*40)
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print("="*40)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_equity['step'], df_equity['capital'], label='Equity Curve')
    plt.title(f'Trading Strategy Evaluation (Final Return: {total_return:.2f}%)')
    plt.xlabel('Trade Step')
    plt.ylabel('Capital (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_plot = "eval_results.png"
    plt.savefig(output_plot)
    print(f"Saved equity curve to {output_plot}")
    
    # Cleanup
    ray.get(rollout_manager.dispose.remote())

if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
