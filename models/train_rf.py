from sklearn.ensemble import RandomForestRegressor
import joblib, polars as pl, os
TICKERS = ["NVDA","AAPL","GOOGL","MSFT","TSLA",
           "ETH-USD","QQQ","SPY","DXY","TLT"]
for sym in TICKERS:
    df = pl.read_parquet(f"data/raw/{sym}.parquet").with_columns(
        ret_next=pl.col("close").pct_change().shift(-1),
        sma20=pl.col("close").rolling_mean(20),
        sma50=pl.col("close").rolling_mean(50),
    ).drop_nulls()
    X, y = df.select(["sma20","sma50"]).to_numpy(), df["ret_next"].to_numpy()
    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X, y)
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, f"outputs/{sym}_rf.joblib")
    print(f"✅ {sym} model saved (R²={model.score(X,y):.3f})")
