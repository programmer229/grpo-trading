# GRPO R1-Trader

A crypto trading agent implementation using **Group Relative Policy Optimization (GRPO)**, inspired by DeepSeek-R1.

## Features
- **Data Layer**: Fetches historical crypto data (BTC-USD) using `yfinance` and formats it into prompts.
- **Trading Environment**: Simulates trades based on model output (`<answer>BUY</answer>`) and calculates rewards based on future price movements.
- **GRPO Trainer**: Implements the GRPO algorithm:
    - Samples $G$ completions per prompt.
    - Normalizes advantages within the group.
    - Optimizes Policy with PPO-style clipping and KL penalty.
- **Model**: Designed for `Qwen/Qwen2.5-0.5B-Instruct` but compatible with any CausalLM.

## Project Structure
```
grpo_trader/
├── data/
│   ├── loader.py       # yfinance data fetching
│   └── processor.py    # Prompt engineering
├── env/
│   └── trading_env.py  # Reward logic
├── model/
│   └── modeling.py     # Model loading
├── train/
│   ├── grpo_trainer.py # GRPO training loop
│   └── loss.py         # GRPO loss function
└── main.py             # Entry point
```

## Installation

This project uses `uv` for dependency management.

1. Clone the project and navigate to the root.
2. Sync dependencies:

```bash
uv sync
source .venv/bin/activate
```

## Usage

```bash
python3 -m grpo_trader.main --ticker BTC-USD --epochs 1 --batch_size 2
```

## Testing

```bash
python3 -m unittest tests/test_grpo.py
```

## Development

When you first clone the repo and you intend to push changes run the following:

```bash
pre-commit install
```

If you want to run pre-commit hooks manually run:

```bash
pre-commit run --all-files
```
