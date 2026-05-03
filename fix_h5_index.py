"""Fix CollectedData h5 files where index.name == 'bodyparts' conflicts with column level name."""
from pathlib import Path
import pandas as pd

LABELED_DATA = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"

fixed = 0
skipped = 0
for h5 in Path(LABELED_DATA).rglob("CollectedData_*.h5"):
    try:
        df = pd.read_hdf(str(h5))
        if df.index.name == "bodyparts":
            df.index.name = None
            df.to_hdf(str(h5), key="df_with_missing", mode="w")
            print(f"Fixed:   {h5.parent.name}/{h5.name}")
            fixed += 1
        else:
            skipped += 1
    except Exception as e:
        print(f"ERROR {h5.parent.name}: {e}")

print(f"\nDone — {fixed} fixed, {skipped} already ok")
