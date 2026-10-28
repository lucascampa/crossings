"""Draw a ~50k-row sample of final_dataset.parquet by whole crossings.

Each crossing appears once per year (2019-2024). Sampling whole crossings keeps
every crossing's history together, so a train/test split by crossing or by year
can't leak the same crossing into both sides. Crossings are stratified by state
and by how many incident-years they have (0, 1, 2+), so the class balance and
the geographic mix match the full file.
"""
import sys
import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
TARGET_ROWS, SEED = 50_000, 42

df = pd.read_parquet(src)
cr = df.groupby("crossingid").agg(rows=("incident", "size"),
                                  inc=("incident", "sum"),
                                  state=("statecode", "first"))
cr["inc_bucket"] = cr["inc"].clip(upper=2)
cr["stratum"] = cr["state"].astype(str) + "_" + cr["inc_bucket"].astype(str)

frac = TARGET_ROWS / len(df)
rng = np.random.default_rng(SEED)
picked = []
for _, g in cr.groupby("stratum"):
    # Randomized rounding so small strata are kept in proportion on average.
    n = len(g) * frac
    n = int(np.floor(n) + (rng.random() < n - np.floor(n)))
    if n:
        picked.extend(rng.choice(g.index.to_numpy(), size=n, replace=False))

out = df[df["crossingid"].isin(set(picked))].sort_values(["crossingid", "asof_year"])
out.reset_index(drop=True).to_parquet(dst, index=False)
print(f"{len(picked)} crossings, {len(out)} rows, incident rate {out.incident.mean():.4%}")
