"""
Custom keypoint labeler for DLC horse joint project.
Usage: python julic_labeler.py <folder_name>

Controls:
  Left-click  — place current keypoint, advance to next
  Right-click — skip current keypoint
  N / D       — next frame
  P / A       — previous frame
  R           — reset all labels on current frame
  S           — save
  Q           — save and quit
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

LABELED_DATA = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"
SCORER = "julic"
BODYPARTS = [
    "l_f_hoof",        "r_f_hoof",
    "l_b_hoof",        "r_b_hoof",
    "l_front_fetlock", "r_front_fetlock",
    "l_hind_fetlock",  "r_hind_fetlock",
    "l_knee",          "r_knee",
    "l_hock",          "r_hock",
]
COLORS = [
    (255,   0, 255), (0,  120, 255),
    (255, 255,   0), (255,  10,   0),
    (255,   0, 255), (0,  120, 255),
    (255, 255,   0), (255,  10,   0),
    (255,   0, 255), (0,  120, 255),
    (255, 255,   0), (255,  10,   0),
]
DISPLAY_MAX_W = 1400
DISPLAY_MAX_H = 900


def build_columns():
    return pd.MultiIndex.from_tuples(
        [(SCORER, bp, c) for bp in BODYPARTS for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"],
    )


def rel_path(folder_name, img_name):
    return f"labeled-data/{folder_name}/{img_name}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python julic_labeler.py <folder_name>")
        sys.exit(1)

    folder = Path(LABELED_DATA) / sys.argv[1]
    if not folder.exists():
        print(f"Folder not found: {folder}")
        sys.exit(1)

    images = sorted(folder.glob("*.png"))
    if not images:
        print(f"No PNG images in {folder}")
        sys.exit(1)

    print(f"Loaded {len(images)} frames from {folder.name}")

    all_cols = build_columns()
    h5_path = folder / f"CollectedData_{SCORER}.h5"

    if h5_path.exists():
        labels = pd.read_hdf(str(h5_path))
        for col in all_cols:
            if col not in labels.columns:
                labels[col] = np.nan
        labels.index.name = None
        print("Loaded existing labels.")
    else:
        idx = [rel_path(folder.name, img.name) for img in images]
        labels = pd.DataFrame(index=idx, columns=all_cols, dtype=float)
        labels.index.name = None

    fi = [0]      # frame index
    bp = [0]      # bodypart index
    zoom = [1.0]  # zoom multiplier
    scale = [1.0]

    def save():
        labels.to_hdf(str(h5_path), key="df_with_missing", mode="w")
        labels.to_csv(str(folder / f"CollectedData_{SCORER}.csv"))
        n_labeled = int((~labels.isnull().all(axis=1)).sum())
        print(f"Saved — {n_labeled}/{len(images)} frames have at least one label")

    def mouse_cb(event, x, y, flags, param):
        rp = rel_path(folder.name, images[fi[0]].name)
        if rp not in labels.index:
            labels.loc[rp] = np.nan
        if event == cv2.EVENT_LBUTTONDOWN:
            labels.loc[rp, (SCORER, BODYPARTS[bp[0]], "x")] = x / scale[0]
            labels.loc[rp, (SCORER, BODYPARTS[bp[0]], "y")] = y / scale[0]
            bp[0] = (bp[0] + 1) % len(BODYPARTS)

    cv2.namedWindow("julic_labeler", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("julic_labeler", mouse_cb)

    while True:
        img = cv2.imread(str(images[fi[0]]))
        h, w = img.shape[:2]
        s = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h) * zoom[0]
        s = max(0.1, min(s, 8.0))
        scale[0] = s
        disp = cv2.resize(img, (int(w * s), int(h * s)))

        rp = rel_path(folder.name, images[fi[0]].name)
        if rp in labels.index:
            for i, part in enumerate(BODYPARTS):
                xv = labels.loc[rp, (SCORER, part, "x")]
                yv = labels.loc[rp, (SCORER, part, "y")]
                if pd.notna(xv) and pd.notna(yv):
                    cx, cy = int(float(xv) * s), int(float(yv) * s)
                    col = COLORS[i]
                    is_current = (i == bp[0])
                    radius = 9 if is_current else 6
                    thickness = 2 if is_current else -1
                    cv2.circle(disp, (cx, cy), radius, col, thickness, cv2.LINE_AA)
                    cv2.putText(disp, part, (cx + 9, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        current_bp = BODYPARTS[bp[0]]
        n_done = sum(
            pd.notna(labels.loc[rp, (SCORER, p, "x")]) if rp in labels.index else False
            for p in BODYPARTS
        )
        status = (f"Frame {fi[0]+1}/{len(images)}  |  "
                  f"Next: {current_bp} ({bp[0]+1}/{len(BODYPARTS)})  |  "
                  f"Labeled: {n_done}/{len(BODYPARTS)}  |  "
                  f"</>=frame  Space=skip  +/-=zoom  R=reset  S=save  Q=quit")
        cv2.putText(disp, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("julic_labeler", disp)
        key = cv2.waitKey(30)
        k = key & 0xFF

        if k == ord('q'):
            save()
            break
        elif k in (ord('>'), ord('.')):  # next frame
            fi[0] = min(fi[0] + 1, len(images) - 1)
            bp[0] = 0
        elif k in (ord('<'), ord(',')):  # prev frame
            fi[0] = max(fi[0] - 1, 0)
            bp[0] = 0
        elif k == 32:  # spacebar — skip keypoint
            bp[0] = (bp[0] + 1) % len(BODYPARTS)
        elif k in (ord('+'), ord('=')):
            zoom[0] = min(zoom[0] * 1.3, 8.0)
        elif k == ord('-'):
            zoom[0] = max(zoom[0] / 1.3, 0.1)
        elif k == ord('s'):
            save()
        elif k == ord('r'):
            if rp in labels.index:
                labels.loc[rp] = np.nan
            bp[0] = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
