import json
import argparse
from grpo_trader.data.loader import fetch_crypto_data, split_data
from grpo_trader.data.processor import CryptoDataset

def generate_jsonl(output_file, ticker="BTC-USD", period="1mo"):
    print(f"Fetching data for {ticker}...")
    df = fetch_crypto_data(ticker, period)
    train_df, _ = split_data(df, train_ratio=1.0) # Use all for this demo
    dataset = CryptoDataset(train_df)
    
    print(f"Generating {len(dataset)} samples...")
    
    with open(output_file, 'w') as f:
        for i in range(len(dataset)):
            item = dataset[i]
            # Slime expects: prompt, label (optional), metadata
            record = {
                "prompt": item['prompt'],
                "label": "", # No ground truth label needed for pure RL, or could be optimal action
                "metadata": {
                    "current_price": item['current_price'],
                    "next_price": item['next_price']
                }
            }
            f.write(json.dumps(record) + "\n")
            
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="train_data.jsonl")
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    args = parser.parse_args()
    
    generate_jsonl(args.output, args.ticker)
