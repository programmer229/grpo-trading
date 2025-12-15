import pandas as pd

def format_market_data_prompt(df_window):
    """
    Formats a window of market data into a text prompt.
    """
    prompt = "Analyze the following crypto market data and decide whether to Buy, Sell, or Hold.\n\n"
    prompt += "Recent Market History:\n"
    
    for i, row in df_window.iterrows():
        date_str = str(row['date']) if 'date' in row else f"T-{len(df_window)-i-1}"
        returns_str = f"{row['returns']*100:+.2f}%" if not pd.isna(row['returns']) else "N/A"
        prompt += f"Time: {date_str} | Price: {row['close']:.2f} ({returns_str}) | SMA5: {row['sma_5']:.2f} | SMA20: {row['sma_20']:.2f}\n"
        
    # Add Technical Analysis Summary
    latest = df_window.iloc[-1]
    
    # RSI Analysis
    rsi_text = f"RSI is {latest['rsi']:.1f}"
    if latest['rsi'] > 70: rsi_text += " (Overbought)"
    elif latest['rsi'] < 30: rsi_text += " (Oversold)"
    else: rsi_text += " (Neutral)"
    
    # MACD Analysis
    macd_text = "MACD is "
    if latest['macd'] > latest['macd_signal']: macd_text += "Bullish (Above Signal)"
    else: macd_text += "Bearish (Below Signal)"
    
    # Bollinger Bands Analysis
    bb_text = "Price is "
    if latest['close'] > latest['bb_upper']: bb_text += "Above Upper Bollinger Band (High Volatility/Possible Reversal)"
    elif latest['close'] < latest['bb_lower']: bb_text += "Below Lower Bollinger Band (Oversold/Possible Reversal)"
    else: bb_text += "Within Bollinger Bands"
    
    prompt += f"\nTechnical Analysis:\n- {rsi_text}\n- {macd_text}\n- {bb_text}\n"
    
    prompt += "\nInstructions:\n"
    prompt += "1. Analyze the trend based on price, moving averages, and technical indicators.\n"
    prompt += "2. Output your reasoning inside <think> tags.\n"
    prompt += "3. Output your final decision (Buy, Sell, or Hold) inside <answer> tags.\n"
    prompt += "Format: <think> reasoning </think> <answer> Action </answer>\n"
    
    # Return as chat format for Slime/SGLang
    return [{"role": "user", "content": prompt}]

class CryptoDataset:
    def __init__(self, df, window_size=10):
        self.df = df
        self.window_size = window_size
        
    def __len__(self):
        return len(self.df) - self.window_size - 1
        
    def __getitem__(self, idx):
        # Window for observation
        window = self.df.iloc[idx : idx + self.window_size]
        
        # Target for reward calculation (next price change)
        current_price = window.iloc[-1]['close']
        next_price = self.df.iloc[idx + self.window_size]['close']
        
        prompt = format_market_data_prompt(window)
        
        return {
            "prompt": prompt,
            "current_price": current_price,
            "next_price": next_price
        }
