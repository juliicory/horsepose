"""
Horse detection test — runs SuperAnimal-Quadruped inference on a video file
with optical flow tracking between keyframes. Press Q to quit.

Usage:
    python horse_detection_test.py
    python horse_detection_test.py horse_vidTest.mp4
    python horse_detection_test.py horse_vidTest.mov --detector-threshold 0.7 --pose-threshold 0.1 --interval 2
"""

import sys
import argparse
import concurrent.futures
import cv2
import numpy as np
import torch
import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
from deeplabcut.pose_estimation_pytorch.apis import (
    get_pose_inference_runner,
    get_detector_inference_runner,
)

ANIMAL_COLORS = [
    (0,   255,   0),   # green
    (0,   128, 255),   # orange
    (255,  64,  64),   # blue
]
DISPLAY_PARTS = ("nose", "hoof", "paw")

# --- OPTICAL FLOW ---
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
_prev_gray = None
_of_points = None
_of_meta   = []


def _horses_to_points(horses):
    points, meta = [], []
    for animal_idx, horse in enumerate(horses):
        for kp in horse:
            points.append([[kp["x"], kp["y"]]])
            meta.append((animal_idx, kp["name"], kp["score"]))
    if points:
        return np.array(points, dtype=np.float32), meta
    return None, []


def _points_to_horses(points, meta):
    if points is None or not meta:
        return []
    n_animals = max(m[0] for m in meta) + 1
    horses = [[] for _ in range(n_animals)]
    for pt, (animal_idx, name, score) in zip(points, meta):
        x, y = pt[0]
        horses[animal_idx].append({
            "name":  name,
            "x":     float(x),
            "y":     float(y),
            "score": float(score),
        })
    return [h for h in horses if h]


def anchor_optical_flow(horses, gray):
    global _prev_gray, _of_points, _of_meta
    _prev_gray = gray

    # Build a lookup of current tracked positions by (animal_idx, name)
    prev_lookup = {}
    if _of_points is not None:
        for pt, (animal_idx, name, _) in zip(_of_points, _of_meta):
            prev_lookup[(animal_idx, name)] = pt[0]

    # For uncertain keypoints, substitute the last tracked position if available
    merged = []
    for animal_idx, horse in enumerate(horses):
        merged_horse = []
        for kp in horse:
            if kp["certain"]:
                merged_horse.append(kp)
            else:
                prev_pos = prev_lookup.get((animal_idx, kp["name"]))
                if prev_pos is not None:
                    merged_horse.append({**kp, "x": float(prev_pos[0]), "y": float(prev_pos[1])})
                # else: no prior position either — skip
        if merged_horse:
            merged.append(merged_horse)

    _of_points, _of_meta = _horses_to_points(merged)


def track_optical_flow(gray):
    global _prev_gray, _of_points
    if _prev_gray is None or _of_points is None or len(_of_points) == 0:
        _prev_gray = gray
        return _points_to_horses(_of_points, _of_meta)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        _prev_gray, gray, _of_points, None, **LK_PARAMS
    )
    _prev_gray = gray
    if new_pts is not None and status is not None:
        for i, (ok, pt) in enumerate(zip(status, new_pts)):
            if ok[0]:
                _of_points[i] = pt
    return _points_to_horses(_of_points, _of_meta)


# --- MODEL ---
def build_runners(detector_threshold, max_individuals, device):
    print(f"Loading model on {device}...")
    config = modelzoo.load_super_animal_config(
        super_animal="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
    )
    config["metadata"]["individuals"] = [f"animal{i}" for i in range(max_individuals)]
    if config.get("detector") is not None:
        config["detector"]["model"]["box_score_thresh"] = detector_threshold

    bodyparts = config["metadata"]["bodyparts"]

    pose_runner = get_pose_inference_runner(
        model_config=config,
        snapshot_path=modelzoo.get_super_animal_snapshot_path(
            dataset="superanimal_quadruped", model_name="hrnet_w32"
        ),
        max_individuals=max_individuals,
        device=device,
    )
    detector_runner = get_detector_inference_runner(
        model_config=config,
        snapshot_path=modelzoo.get_super_animal_snapshot_path(
            dataset="superanimal_quadruped", model_name="fasterrcnn_resnet50_fpn_v2"
        ),
        max_individuals=max_individuals,
        device=device,
    )
    print("Model ready.")
    return pose_runner, detector_runner, bodyparts


def run_inference(frame, pose_runner, detector_runner, bodyparts, score_threshold):
    bbox_preds = detector_runner.inference([frame])
    if not bbox_preds:
        return []
    pose_preds = pose_runner.inference([(frame, bbox_preds[0])])
    if not pose_preds:
        return []
    poses = pose_preds[0].get("bodyparts", None)
    if poses is None:
        return []
    horses = []
    for individual in poses:
        keypoints = []
        for bp_idx, (x, y, score) in enumerate(individual):
            keypoints.append({
                "name":    bodyparts[bp_idx],
                "x":       float(x),
                "y":       float(y),
                "score":   float(score),
                "certain": score >= score_threshold,  # flag — False means keep old position
            })
        if any(kp["certain"] for kp in keypoints):
            horses.append(keypoints)
    return horses


def draw_horses(frame, horses):
    for animal_idx, horse in enumerate(horses):
        color = ANIMAL_COLORS[animal_idx % len(ANIMAL_COLORS)]
        for kp in horse:
            if not any(part in kp["name"] for part in DISPLAY_PARTS):
                continue
            cx, cy = int(kp["x"]), int(kp["y"])
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.putText(frame, kp["name"], (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="horse_vidTest.mov",
                        help="Path to video file (default: horse_vidTest.mov)")
    parser.add_argument("--detector-threshold", type=float, default=0.6,
                        help="Confidence for animal bounding box detection (default 0.6)")
    parser.add_argument("--pose-threshold",     type=float, default=0.5,
                        help="Confidence for individual keypoints (default 0.2)")
    parser.add_argument("--interval",           type=int,   default=4,
                        help="DLC inference every N frames; optical flow in between")
    parser.add_argument("--max-individuals",    type=int,   default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_runner, detector_runner, bodyparts = build_runners(
        args.detector_threshold, args.max_individuals, device
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video: {args.video}  {w}x{h}  {fps:.1f}fps")
    print(f"Inference every {args.interval} frames (detector>{args.detector_threshold}, pose>{args.pose_threshold}). Press Q to quit.")

    frame_count      = 0
    horses           = []
    executor         = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inference_future = None
    inference_gray   = None

    display_w = w // 1
    display_h = h // 1

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pick up finished inference and re-anchor optical flow
        if inference_future is not None and inference_future.done():
            try:
                new_horses = inference_future.result()
                if new_horses:
                    anchor_optical_flow(new_horses, inference_gray)
                    horses = new_horses
            except Exception as e:
                print(f"Inference error frame {frame_count}: {e}")
            inference_future = None

        # Fire inference in background every interval frames
        if inference_future is None and frame_count % args.interval == 1:
            inference_gray   = gray.copy()
            inference_future = executor.submit(
                run_inference, frame.copy(), pose_runner, detector_runner,
                bodyparts, args.pose_threshold
            )
            if frame_count == 1:
                print("Waiting for initial detection...")

        # Optical flow tracks every frame while inference runs in background
        horses = track_optical_flow(gray)

        draw_horses(frame, horses)

        status = f"Frame {frame_count} | {len(horses)} horse(s) | {device.upper()}"
        cv2.putText(frame, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Horse Detection Test", cv2.resize(frame, (display_w, display_h)))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
