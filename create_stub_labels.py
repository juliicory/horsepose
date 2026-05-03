import pandas as pd
from pathlib import Path

LABELED_DATA = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"
SCORER = "julic"
BODYPARTS = [
    "l_f_hoof", "r_f_hoof", "l_b_hoof", "r_b_hoof",
    "l_front_fetlock", "r_front_fetlock", "l_hind_fetlock", "r_hind_fetlock",
    "l_knee", "r_knee", "l_hock", "r_hock",
]
FOLDERS = ["trot_side", "canter_graham", "short_trot_Ben"]

all_cols = pd.MultiIndex.from_tuples(
    [(SCORER, bp, c) for bp in BODYPARTS for c in ("x", "y")],
    names=["scorer", "bodyparts", "coords"],
)

for name in FOLDERS:
    folder = Path(LABELED_DATA) / name
    h5 = folder / f"CollectedData_{SCORER}.h5"
    if h5.exists():
        print(f"Already exists: {name}")
        continue
    images = sorted(folder.glob("*.png"))
    if not images:
        print(f"No images found: {name}")
        continue
    index = [f"labeled-data/{name}/{img.name}" for img in images]
    df = pd.DataFrame(index=index, columns=all_cols, dtype=float)
    df.index.name = None
    df.to_hdf(str(h5), key="df_with_missing", mode="w")
    print(f"Created: {name} ({len(images)} frames)")
