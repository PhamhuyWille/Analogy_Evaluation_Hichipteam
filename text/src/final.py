from pathlib import Path
import pandas as pd

ROOT = Path("results")

tcc = pd.read_csv(ROOT / "tcc" / "test_TCC_scores.csv")
ms = pd.read_csv(ROOT / "ms" / "test_MS_scores.csv")
m = pd.read_csv(ROOT / "m" / "test_M_scores.csv")

assert len(tcc) == len(ms) == len(m), "The three score files have different numbers of rows."

# Use scalar zeros; pandas broadcasts them to every row.
submission = pd.DataFrame({
    "id": tcc["id"],
    "TCC": tcc["score"].round().astype(int),
    "MS": ms["score"].round().astype(int),
    "M": m["score"].round().astype(int),
    "VC": 0,
    "VA": 0,
    "VE": 0
})

submission.to_csv(ROOT / "submission.csv", index=False)

print("submission.csv saved!")
print(submission.head())
