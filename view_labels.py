"""
View labeled frames from the DLC_Horse imported dataset.

Usage:
    python view_labels.py <name> <frame>

    name   — substring to match against labeled-data folder names (case-insensitive)
    frame  — filename (img0000.png) or numeric index (0, 1, -1 for last, etc.)

Examples:
    python view_labels.py "bob walk" img0000.png
    python view_labels.py "bob walk" -1
    python view_labels.py "20210201 annie walk_0" img0010.png
"""

import sys
import os
import glob
import pandas as pd
import cv2

LABELED_DATA = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"


def find_folder(name):
    # Search one level deep (flat) and two levels deep (letter subfolders)
    candidates = []
    for entry in os.listdir(LABELED_DATA):
        entry_path = os.path.join(LABELED_DATA, entry)
        if os.path.isdir(entry_path):
            if len(entry) == 1 and entry.isalpha():
                for sub in os.listdir(entry_path):
                    if name.lower() in sub.lower():
                        candidates.append(os.path.join(entry_path, sub))
            elif name.lower() in entry.lower():
                candidates.append(entry_path)
    matches = [c for c in candidates if os.path.isdir(c)]
    if not matches:
        print(f"No folder matching '{name}' found.")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple matches for '{name}', pick one:")
        for d in matches:
            print(f"  {os.path.relpath(d, LABELED_DATA)}")
        sys.exit(1)
    return matches[0]


def show_frame(folder, frame_idx):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print(f"No CSV found in {folder}")
        sys.exit(1)
    csv = csv_files[0]

    df = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    imgs = sorted(glob.glob(os.path.join(folder, "*.png")))

    if not imgs:
        print(f"No PNG images found in {folder}")
        sys.exit(1)

    img_names = [os.path.basename(p) for p in imgs]
    if frame_idx in img_names:
        idx = img_names.index(frame_idx)
    else:
        idx = int(frame_idx)
        if idx >= len(imgs) or idx < -len(imgs):
            print(f"Frame '{frame_idx}' not found. Available: {img_names[0]} … {img_names[-1]}")
            sys.exit(1)

    img_path = imgs[idx]
    frame = cv2.imread(img_path)
    coords = df.iloc[idx]
    scorer = df.columns.get_level_values(0)[0]

    drawn = 0
    for bp in df.columns.get_level_values(1).unique():
        try:
            x = coords[(scorer, bp, "x")]
            y = coords[(scorer, bp, "y")]
            if pd.isna(x) or pd.isna(y):
                continue
            cx, cy = int(x), int(y)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.putText(frame, bp, (cx + 4, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            drawn += 1
        except (KeyError, ValueError):
            pass

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (w // 2, h // 2))

    name = os.path.basename(folder)
    title = f"{name}  |  frame {frame_idx} ({os.path.basename(img_path)})  |  {drawn} keypoints"
    print(title)
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    name = sys.argv[1]
    frame_idx = sys.argv[2]
    folder = find_folder(name)
    show_frame(folder, frame_idx)
