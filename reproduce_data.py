import json
import os
import sys
from unittest.mock import MagicMock

# Mock yfinance and pandas
sys.modules["yfinance"] = MagicMock()
sys.modules["pandas"] = MagicMock()

# Mock loader and processor
import pandas as pd
class MockDataFrame:
    def __init__(self, data=None):
        self._data = data or []
    def __len__(self):
        return len(self._data)
    
    class IlocIndexer:
        def __init__(self, parent):
            self.parent = parent
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return MockDataFrame(self.parent._data[idx])
            return self.parent._data[idx]

    @property
    def iloc(self):
        return self.IlocIndexer(self)
    def iterrows(self):
        for i, row in enumerate(self._data):
            yield i, row

# Mock fetch_crypto_data
def mock_fetch_crypto_data(ticker, period):
    data = []
    for i in range(100):
        data.append({
            "date": f"2025-01-{i%30+1}",
            "close": 100.0 + i,
            "sma_5": 100.0,
            "sma_20": 100.0,
            "volatility": 1.0
        })
    return MockDataFrame(data)

# Mock split_data
def mock_split_data(df, train_ratio=0.8):
    return df, df # Return same for simplicity

# Patch loader
import grpo_trader.data.loader
grpo_trader.data.loader.fetch_crypto_data = mock_fetch_crypto_data
grpo_trader.data.loader.split_data = mock_split_data

# Patch processor (pandas dependency)
import grpo_trader.data.processor
grpo_trader.data.processor.pd = MagicMock()
grpo_trader.data.processor.pd.DataFrame = MockDataFrame

# Run generation
from grpo_trader.slime_adapter.gen_data import generate_jsonl

print("Generating dummy data...")
generate_jsonl(".", "BTC-USD")

print("\nChecking train_data.jsonl content:")
with open("train_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        if i < 3:
            print(f"Line {i}: {line.strip()}")
        else:
            break
