"""
detection_trainer.py

Phase 1a — create_project(): create DLC project + extract frames from ref_vids (run once)
Phase 1b — import_dlc_horse(): import DLC_Horse predictions as training labels (run once)
Phase 2a — dataset(): create training dataset
Phase 2b — train(): train / resume the joint model
"""

import os

PROJECT_NAME  = "horse_joints"
EXPERIMENTER  = "julic"
WORKING_DIR   = r"C:\Users\julic\Documents\GitHub\horsepose"

VIDEOS = [
    r"C:\Users\julic\Documents\GitHub\horsepose\ref_vids\trot_side.mp4",
    r"C:\Users\julic\Documents\GitHub\horsepose\ref_vids\canter_slomo.mp4",
    r"C:\Users\julic\Documents\GitHub\horsepose\ref_vids\canter_graham.mov",
    r"C:\Users\julic\Documents\GitHub\horsepose\ref_vids\short_trot_Ben.MOV",
]


# =============================================================================
# PHASE 1a — Create DLC project + extract frames (run once)
# =============================================================================
def create_project():
    import deeplabcut
    import ruamel.yaml

    config_path = deeplabcut.create_new_project(
        PROJECT_NAME,
        EXPERIMENTER,
        VIDEOS,
        working_directory=WORKING_DIR,
        copy_videos=False,
    )
    print(f"\nProject created.\nConfig: {config_path}\n")

    yaml = ruamel.yaml.YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)

    cfg["bodyparts"] = [
        "l_f_hoof",        "r_f_hoof",
        "l_b_hoof",        "r_b_hoof",
        "l_front_fetlock", "r_front_fetlock",
        "l_hind_fetlock",  "r_hind_fetlock",
        "l_knee",          "r_knee",
        "l_hock",          "r_hock",
    ]
    cfg["skeleton"] = [
        ["l_f_hoof",       "l_front_fetlock"],
        ["l_front_fetlock","l_knee"],
        ["r_f_hoof",       "r_front_fetlock"],
        ["r_front_fetlock","r_knee"],
        ["l_b_hoof",       "l_hind_fetlock"],
        ["l_hind_fetlock", "l_hock"],
        ["r_b_hoof",       "r_hind_fetlock"],
        ["r_hind_fetlock", "r_hock"],
    ]
    cfg["numframes2pick"] = 40

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print("Bodyparts set to 12 joint keypoints.")
    print("\nExtracting frames (kmeans, ~1 min per video)...")
    deeplabcut.extract_frames(config_path, mode="automatic", algo="kmeans", userfeedback=False)
    print(f"\n=== Phase 1a done ===\nConfig path: {config_path}")


# =============================================================================
# PHASE 1b — Import DLC_Horse predictions as training labels
# =============================================================================
BODYPART_MAP = {
    "LeftFrontHoof":     "l_f_hoof",
    "RightFrontHoof":    "r_f_hoof",
    "LeftHindHoof":      "l_b_hoof",
    "RightHindHoof":     "r_b_hoof",
    "LeftFrontFetlock":  "l_front_fetlock",
    "RightFrontFetlock": "r_front_fetlock",
    "LeftHindFetlock":   "l_hind_fetlock",
    "RightHindFetlock":  "r_hind_fetlock",
    "LeftKnee":          "l_knee",
    "RightKnee":         "r_knee",
    "LeftHock":          "l_hock",
    "RightHock":         "r_hock",
}

DLC_HORSE_DIR = r"C:\Users\julic\Documents\GitHub\DLC_Horse\dataset"
CONFIG_PATH   = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\config.yaml"
LABELED_DATA  = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\labeled-data"
SCORER        = EXPERIMENTER

LIKELIHOOD_THRESHOLD = 0.6
FRAME_SAMPLE_STEP    = 5


def import_dlc_horse():
    import pandas as pd
    import numpy as np
    import cv2
    from pathlib import Path

    OUR_BODYPARTS = [
        "l_f_hoof",        "r_f_hoof",
        "l_b_hoof",        "r_b_hoof",
        "l_front_fetlock", "r_front_fetlock",
        "l_hind_fetlock",  "r_hind_fetlock",
        "l_knee",          "r_knee",
        "l_hock",          "r_hock",
    ]

    h5_files = [f for f in os.listdir(DLC_HORSE_DIR) if f.endswith(".h5")]
    print(f"Found {len(h5_files)} .h5 prediction files in DLC_Horse dataset")

    for h5_file in h5_files:
        h5_path = os.path.join(DLC_HORSE_DIR, h5_file)
        df = pd.read_hdf(h5_path)

        if df.columns.nlevels == 3:
            df.columns = df.columns.droplevel(0)

        if h5_file == h5_files[0]:
            bodyparts_found = df.columns.get_level_values(0).unique().tolist()
            print(f"Bodyparts in DLC_Horse: {bodyparts_found}")
            mappable = [b for b in bodyparts_found if b in BODYPART_MAP]
            print(f"Mappable to our project: {mappable}")

        base = h5_file.split("DLC_resnet")[0].rstrip("_")
        avi_path = os.path.join(DLC_HORSE_DIR, base + ".avi")
        if not os.path.exists(avi_path):
            candidates = [f for f in os.listdir(DLC_HORSE_DIR)
                          if f.lower().endswith(".avi") and base.lower() in f.lower()]
            if not candidates:
                print(f"  No .avi found for {h5_file}, skipping")
                continue
            avi_path = os.path.join(DLC_HORSE_DIR, candidates[0])

        video_name = Path(avi_path).stem
        out_dir = os.path.join(LABELED_DATA, video_name)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(avi_path)
        frame_indices = df.index.tolist()[::FRAME_SAMPLE_STEP]

        rows = []
        saved = 0
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame_img = cap.read()
            if not ret:
                continue

            row = {}
            for dlc_bp, our_bp in BODYPART_MAP.items():
                try:
                    x    = df.loc[frame_idx, (dlc_bp, "x")]
                    y    = df.loc[frame_idx, (dlc_bp, "y")]
                    lkly = df.loc[frame_idx, (dlc_bp, "likelihood")]
                except KeyError:
                    continue
                if lkly < LIKELIHOOD_THRESHOLD or np.isnan(x) or np.isnan(y):
                    continue
                row[(our_bp, "x")] = round(float(x), 2)
                row[(our_bp, "y")] = round(float(y), 2)

            if not row:
                continue

            img_name = f"img{int(frame_idx):04d}.png"
            img_path = os.path.join(out_dir, img_name)
            cv2.imwrite(img_path, frame_img)

            rel_path = f"labeled-data/{video_name}/{img_name}"
            rows.append((rel_path, row))
            saved += 1

        cap.release()
        print(f"  {video_name}: {saved} frames saved")

        if not rows:
            continue

        index = [r[0] for r in rows]
        all_cols = pd.MultiIndex.from_tuples(
            [(SCORER, bp, coord)
             for bp in OUR_BODYPARTS
             for coord in ("x", "y")],
            names=["scorer", "bodyparts", "coords"],
        )
        data = pd.DataFrame(index=index, columns=all_cols, dtype=float)
        data.index.name = None

        for rel_path, row in rows:
            for (bp, coord), val in row.items():
                data.loc[rel_path, (SCORER, bp, coord)] = val

        csv_out = os.path.join(out_dir, f"CollectedData_{SCORER}.csv")
        h5_out  = os.path.join(out_dir, f"CollectedData_{SCORER}.h5")
        data.to_csv(csv_out)
        data.to_hdf(h5_out, key="df_with_missing", mode="w")
        print(f"    → {csv_out}")

    print("\n=== Phase 1b done ===")


# =============================================================================
# PHASE 2a — Create training dataset
# =============================================================================
def _detect_448_shuffle():
    import glob
    import yaml
    import re
    configs = sorted(
        glob.glob(
            CONFIG_PATH.replace("config.yaml", "") +
            "dlc-models-pytorch/iteration-0/*/train/pytorch_config.yaml"
        ),
        key=lambda p: int(re.search(r"shuffle(\d+)", p).group(1))
    )
    for cfg_path in reversed(configs):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        w = cfg.get("data", {}).get("train", {}).get("crop_sampling", {}).get("width", 448)
        if w == 448:
            m = re.search(r"shuffle(\d+)", cfg_path)
            if m:
                return int(m.group(1))
    return None


def create_dataset():
    import deeplabcut
    import glob
    import re

    print("Creating training dataset (448×448, hrnet_w32 + SA init)...")
    deeplabcut.create_training_dataset(
        CONFIG_PATH,
        net_type="hrnet_w32",
        shuffle=9,
        userfeedback=False,
    )
    configs = sorted(glob.glob(
        CONFIG_PATH.replace("config.yaml", "") +
        "dlc-models-pytorch/iteration-0/*/train/pytorch_config.yaml"
    ), key=lambda p: int(re.search(r"shuffle(\d+)", p).group(1)))
    m = re.search(r"shuffle(\d+)", configs[-1])
    shuffle_num = int(m.group(1)) if m else "?"
    print(f"\n=== Phase 2a done (shuffle {shuffle_num}) ===")
    print("Now run: python detection_trainer.py train")


# =============================================================================
# PHASE 2b — Train / resume
# =============================================================================
def _find_latest_snapshot(shuffle_num):
    import glob
    import re
    train_dir = CONFIG_PATH.replace("config.yaml", "") + \
        f"dlc-models-pytorch/iteration-0/horse_jointsApr29-trainset95shuffle{shuffle_num}/train"
    snapshots = glob.glob(os.path.join(train_dir, "snapshot*.pt"))
    if not snapshots:
        return None
    # Sort by epoch number embedded in filename
    snapshots.sort(key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    return snapshots[-1]


def train():
    import deeplabcut

    shuffle_num = _detect_448_shuffle()
    if shuffle_num is None:
        print("No 448×448 shuffle found. Run: python detection_trainer.py dataset")
        return

    latest = _find_latest_snapshot(shuffle_num)
    if latest:
        print(f"Resuming from: {os.path.basename(latest)}")
    else:
        print("No existing snapshot — starting from SA pretrained weights.")

    print(f"Training shuffle {shuffle_num} at 448×448...")
    deeplabcut.train_network(
        CONFIG_PATH,
        shuffle=shuffle_num,
        displayiters=100,
        epochs=200,
        gputouse=0,
        snapshot_path=latest,
    )
    print(f"\n=== Phase 2b done ===")
    print(f"Evaluate with: deeplabcut.evaluate_network(CONFIG_PATH, shuffle={shuffle_num})")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "create":
        create_project()
    elif cmd == "import":
        import_dlc_horse()
    elif cmd == "dataset":
        create_dataset()
    elif cmd == "train":
        train()
    else:
        print("Usage:")
        print("  python detection_trainer.py create   — Phase 1a: create project + extract frames")
        print("  python detection_trainer.py import   — Phase 1b: import DLC_Horse labels")
        print("  python detection_trainer.py dataset  — Phase 2a: create training dataset")
        print("  python detection_trainer.py train    — Phase 2b: train / resume")
