# jsonl_to_csv_utf8sig.py
import pandas as pd, io, sys

inp = sys.argv[1] if len(sys.argv) > 1 else "data/train_v2.jsonl"
out = sys.argv[2] if len(sys.argv) > 2 else "data/train_v2.csv"

with io.open(inp, "r", encoding="utf-8") as f:
    df = pd.read_json(f, lines=True)

cols = [c for c in ["instruction","input","output","topic","severity"] if c in df.columns]
df = df[cols]

# Excel-friendly UTF-8 with BOM
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Wrote {out} ({len(df)} rows)")
