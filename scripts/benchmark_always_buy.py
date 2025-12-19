
import json
import argparse
# import pandas as pd
# import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, default="test_data.jsonl", help="Path to test.jsonl")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=2.0)
    args = parser.parse_args()

    print(f"Loading data from {args.test_data}...")
    data = []
    with open(args.test_data, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples.")

    # Sort data by timestamp
    import re
    print("Sorting data by timestamp...")
    def get_timestamp(item):
        prompt_text = str(item.get('prompt', ''))
        matches = re.findall(r"Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", prompt_text)
        if matches:
            return matches[-1]
        return "0000-00-00 00:00:00"

    data.sort(key=get_timestamp)
    print("Data sorted.")

    print("Simulating Baselines...")

    strategies = ["ALWAYS_BUY", "ALWAYS_SELL"]
    results = {}

    for strat in strategies:
        capital = args.initial_capital
        equity_curve = []
        risk_pct = args.risk_per_trade_pct
        
        for i, d in enumerate(data):
            metadata = d.get('metadata', {})
            current_price = metadata.get('current_price')
            next_price = metadata.get('next_price')

            if current_price is None or next_price is None:
                continue

            price_change_pct = (next_price - current_price) / current_price
            position_size = capital * risk_pct
            
            pnl = 0.0
            if strat == "ALWAYS_BUY":
                pnl = position_size * price_change_pct
            elif strat == "ALWAYS_SELL":
                pnl = position_size * (-price_change_pct)
            
            capital += pnl
            
            equity_curve.append({
                'step': i, 'capital': capital,
                'price_change': price_change_pct, 'pnl': pnl
            })

        if not equity_curve:
            print(f"No valid data points found for {strat}.")
            continue

        final_capital = equity_curve[-1]['capital']
        total_return = (final_capital - args.initial_capital) / args.initial_capital * 100
        results[strat] = {
            "final_capital": final_capital,
            "return": total_return
        }

    print("\n--- Baseline Results ---")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    for strat, res in results.items():
        print(f"{strat:12} | Final: ${res['final_capital']:,.2f} | Return: {res['return']:.2f}%")
    print("------------------------\n")

if __name__ == "__main__":
    main()
