import yfinance as yf
import pandas as pd
import numpy as np

def fetch_crypto_data(ticker="BTC-USD", period="1mo", interval="1h"):
    """
    Fetches crypto data from yfinance.
    """
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Ensure columns are flat if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Calculate some basic indicators for the model to use
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['volatility'] = df['returns'].rolling(window=10).std()

    # --- Advanced Technical Indicators ---
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    df['bb_upper'] = df['sma_20'] + (df['close'].rolling(window=20).std() * 2)
    df['bb_lower'] = df['sma_20'] - (df['close'].rolling(window=20).std() * 2)

    # Volume Trend (SMA 20)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    
    df = df.dropna()
    return df

def split_data(df, train_ratio=0.8):
    """
    Splits data into train and test sets.
    """
    n = len(df)
    train_size = int(n * train_ratio)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df
