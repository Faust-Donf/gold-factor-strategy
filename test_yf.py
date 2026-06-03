from gold_strategy.data.yahoo_loader import load_yahoo_data
from datetime import date
df = load_yahoo_data(["GLD", "SPY", "^VIX", "UUP", "USO", "TLT", "IEF"], date(2023, 1, 1))
print("Downloaded size:", len(df))
print(df.head())
