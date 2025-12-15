import yfinance as yf
import json

def test_news():
    ticker = "BTC-USD"
    t = yf.Ticker(ticker)
    news = t.news
    print(f"--- News for {ticker} ---")
    print(json.dumps(news, indent=2))

if __name__ == "__main__":
    test_news()
