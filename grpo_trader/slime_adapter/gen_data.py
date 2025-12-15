import json
import argparse
from grpo_trader.data.loader import fetch_crypto_data, split_data
from grpo_trader.data.processor import CryptoDataset

def generate_jsonl(output_dir, ticker="BTC-USD", period="1y"):
    print(f"Fetching data for {ticker}...")
    df = fetch_crypto_data(ticker, period)
    
    # Split data
    train_df, test_df = split_data(df, train_ratio=0.8)
    
    datasets = {
        "train": CryptoDataset(train_df),
        "test": CryptoDataset(test_df)
    }

    for split, dataset in datasets.items():
        output_file = f"{output_dir}/{split}_data.jsonl"
        print(f"Generating {len(dataset)} samples for {split}...")
        records = []
        for i in range(len(dataset)):
            item = dataset[i]
            record = {
                "prompt": item['prompt'],
                "label": "",
                "metadata": {
                    "current_price": item['current_price'],
                    "next_price": item['next_price']
                }
            }
            records.append(record)

        import random
        random.shuffle(records)

        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"Saved {len(records)} shuffled samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--period", type=str, default="1y")
    args = parser.parse_args()

    generate_jsonl(args.output_dir, args.ticker, args.period)
