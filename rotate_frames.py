"""
Rotate extracted frames and transform existing labels for a labeled-data folder.
Usage: python rotate_frames.py <folder_name> [cw|ccw|180]

Default direction: cw (90° clockwise — corrects a video recorded 90° CCW).
Transforms label coordinates to match the rotated images, then rotates PNGs in-place.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

LABELED_DATA = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"
SCORER = "julic"

ROTATIONS = {
    "cw":  cv2.ROTATE_90_CLOCKWISE,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
}


def transform_coords(x, y, W, H, direction):
    """Transform (x, y) pixel coords to match rotated image."""
    if direction == "cw":
        return H - 1 - y, x          # new image is H wide, W tall
    elif direction == "ccw":
        return y, W - 1 - x          # new image is H wide, W tall
    elif direction == "180":
        return W - 1 - x, H - 1 - y  # same dimensions
    return x, y


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    folder_name = sys.argv[1]
    direction   = sys.argv[2] if len(sys.argv) > 2 else "cw"

    if direction not in ROTATIONS:
        print(f"Unknown direction '{direction}'. Use: cw, ccw, 180")
        sys.exit(1)

    folder = Path(LABELED_DATA) / folder_name
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    images = sorted(folder.glob("*.png"))
    if not images:
        print(f"No PNG images in {folder}")
        sys.exit(1)

    # Get original image dimensions from first image
    sample = cv2.imread(str(images[0]))
    if sample is None:
        print(f"Could not read {images[0]}")
        sys.exit(1)
    H, W = sample.shape[:2]
    print(f"Original frame size: {W}x{H}")

    rot_code = ROTATIONS[direction]

    # --- Transform existing labels ---
    h5_path = folder / f"CollectedData_{SCORER}.h5"
    if h5_path.exists():
        print(f"Transforming labels in {h5_path.name}...")
        df = pd.read_hdf(str(h5_path))
        df.index.name = None

        bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
        n_transformed = 0

        for bp in bodyparts:
            try:
                xs = df[(SCORER, bp, "x")]
                ys = df[(SCORER, bp, "y")]
            except KeyError:
                continue

            valid = xs.notna() & ys.notna()
            for idx in df.index[valid]:
                ox, oy = float(xs[idx]), float(ys[idx])
                nx, ny = transform_coords(ox, oy, W, H, direction)
                df.loc[idx, (SCORER, bp, "x")] = nx
                df.loc[idx, (SCORER, bp, "y")] = ny
                n_transformed += 1

        df.to_hdf(str(h5_path), key="df_with_missing", mode="w")
        df.to_csv(str(folder / f"CollectedData_{SCORER}.csv"))
        print(f"  Transformed {n_transformed} keypoints.")
    else:
        print("No label file found — only rotating images.")

    # --- Rotate images ---
    print(f"Rotating {len(images)} frames {direction}...")
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read {img_path.name}, skipping")
            continue
        rotated = cv2.rotate(img, rot_code)
        cv2.imwrite(str(img_path), rotated)

    new_W, new_H = (H, W) if direction in ("cw", "ccw") else (W, H)
    print(f"Done — {len(images)} frames rotated {direction}. New size: {new_W}x{new_H}")
    print(f"\nResume labeling with: python julic_labeler.py {folder_name}")


if __name__ == "__main__":
    main()
