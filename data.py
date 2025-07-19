import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

tickers = ["REPYF", "BP", "SHEL", "TTE", "EQNR", "E"]
for t in tickers:
    yt = yf.Ticker(t)
    df = yt.history(period="10y", interval="1d")
    if df.empty:
        print(f"⚠️ No data found for {t}")
    else:
        df.to_csv(f"data/{t}.csv")
        print(f"✅ Saved: data/{t}.csv")
