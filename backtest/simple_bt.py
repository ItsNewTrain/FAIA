import joblib, polars as pl, os
TICKERS = ["NVDA","AAPL","GOOGL","MSFT","TSLA",
           "ETH-USD","QQQ","SPY","DXY","TLT"]
for sym in TICKERS:
    df = pl.read_parquet(f"data/raw/{sym}.parquet").with_columns(
        sma20=pl.col("close").rolling_mean(20),
        sma50=pl.col("close").rolling_mean(50),
    ).drop_nulls()
    model = joblib.load(f"outputs/{sym}_rf.joblib")
    preds = model.predict(df.select(["sma20","sma50"]).to_numpy())
    df = df.with_columns(pred=preds)
    os.makedirs("backtest", exist_ok=True)
    df.write_parquet(f"backtest/{sym}_bt.parquet")
    print(f"✅ {sym} back-test saved.")
