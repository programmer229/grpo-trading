import argparse
import torch
from torch.utils.data import DataLoader
from grpo_trader.data.loader import fetch_crypto_data, split_data
from grpo_trader.data.processor import CryptoDataset
from grpo_trader.env.trading_env import TradingEnvironment
from grpo_trader.model.modeling import load_model_and_tokenizer
from grpo_trader.train.grpo_trainer import GRPOTrainer

def main():
    parser = argparse.ArgumentParser(description="GRPO Trader Training")
    parser.add_argument("--ticker", type=str, default="BTC-USD", help="Crypto ticker")
    parser.add_argument("--period", type=str, default="1mo", help="Data period")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--group_size", type=int, default=4, help="Group size for GRPO")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    try:
        df = fetch_crypto_data(args.ticker, args.period)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    train_df, test_df = split_data(df)
    train_dataset = CryptoDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Training data size: {len(train_dataset)}")
    
    # 2. Load Model
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model.to(device)
    
    # 3. Setup Environment & Trainer
    env = TradingEnvironment()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        env=env,
        optimizer=optimizer,
        group_size=args.group_size,
        device=device
    )
    
    # 4. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            loss = trainer.train_step(batch)
            total_loss += loss
            print(f"Batch Loss: {loss:.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f}")
        
    print("Training complete.")
    
    # Save model
    model.save_pretrained("grpo_trader_model")
    tokenizer.save_pretrained("grpo_trader_model")
    print("Model saved to grpo_trader_model")

if __name__ == "__main__":
    main()
