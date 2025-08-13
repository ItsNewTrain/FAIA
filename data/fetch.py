import yfinance as yf, polars as pl, datetime as dt, os
TICKERS = {
    "NVDA":"NVDA","AAPL":"AAPL","GOOGL":"GOOGL","MSFT":"MSFT","TSLA":"TSLA",
    "ETH-USD":"ETH-USD","QQQ":"QQQ","SPY":"SPY","DXY":"DX-Y.NYB","TLT":"TLT"
}
START = dt.datetime.today() - dt.timedelta(days=3*365)
os.makedirs("data/raw", exist_ok=True)
for short, yahoo in TICKERS.items():
    df = yf.download(yahoo, start=START, interval="1d").reset_index()
    df.columns = [c.lower() for c in df.columns]
    pl.from_pandas(df).write_parquet(f"data/raw/{short}.parquet")
print("✅ Download complete.")
