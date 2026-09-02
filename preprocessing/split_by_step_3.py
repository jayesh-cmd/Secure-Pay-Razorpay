import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
inp      = ROOT / "features.parquet"
out_train = ROOT / "splitted_DS" / "train.parquet"
out_val   = ROOT / "splitted_DS" / "val.parquet"
out_test  = ROOT / "splitted_DS" / "test.parquet"

df = pd.read_parquet(inp)

min_step, max_step = int(df.step.min()), int(df.step.max())
train_cut = min_step + int(0.70*(max_step-min_step))
val_cut   = min_step + int(0.85*(max_step-min_step))

train = df[df.step<=train_cut]
val = df[(df.step>train_cut) & (df.step<=val_cut)]
test  = df[df.step > val_cut]

train.to_parquet(out_train, index=False)
val.to_parquet(out_val, index=False)
test.to_parquet(out_test, index=False)

print("Ranges:",
      f"train [{train.step.min()}..{train.step.max()}]  n={len(train)}",
      f"val   [{val.step.min()}..{val.step.max()}]  n={len(val)}",
      f"test  [{test.step.min()}..{test.step.max()}] n={len(test)}",
      sep="\n")
