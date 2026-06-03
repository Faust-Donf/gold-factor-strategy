import pandas as pd
df = pd.read_csv("https://stooq.com/q/d/l/?s=GLD.US&i=d")
print("Stooq download length:", len(df))
print(df.head())
