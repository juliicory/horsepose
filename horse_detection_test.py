"""
Horse detection test — SuperAnimal-Quadruped inference with optical flow tracking.
Displays body outline + full leg chain (thai → knee → paw). Press Q to quit.

Usage:
    python horse_detection_test.py
    python horse_detection_test.py 2
    python horse_detection_test.py 3 --detector-threshold 0.4 --interval 2
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
from visual_constants import (
    VIDEO_TRANSFORMS, KP_RADIUS, TEXT_SCALE, LINE_THICKNESS,
    INFERENCE_INTERVAL, DETECTOR_THRESHOLD, POSE_THRESHOLD, get_limb_color,
    SHOW_BBOX, BBOX_COLOR,
)

# SA keypoint substrings to display (case-insensitive match against full name)
DISPLAY_PARTS = ("nose", "throat", "withers", "tailbase", "thai", "knee", "paw")

SKELETON = [
    # Body outline
    ("nose",              "throat_base"),
    ("throat_base",       "back_base"),
    ("back_base",         "back_end"),
    ("back_end",          "tail_base"),
    # Front legs
    ("back_base",         "front_left_thai"),
    ("front_left_thai",   "front_left_knee"),
    ("front_left_knee",   "front_left_paw"),
    ("back_base",         "front_right_thai"),
    ("front_right_thai",  "front_right_knee"),
    ("front_right_knee",  "front_right_paw"),
    # Hind legs
    ("back_end",          "back_left_thai"),
    ("back_left_thai",    "back_left_knee"),
    ("back_left_knee",    "back_left_paw"),
    ("back_end",          "back_right_thai"),
    ("back_right_thai",   "back_right_knee"),
    ("back_right_knee",   "back_right_paw"),
]

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
    prev_lookup = {}
    if _of_points is not None:
        for pt, (animal_idx, name, _) in zip(_of_points, _of_meta):
            prev_lookup[(animal_idx, name)] = pt[0]
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
    print(f"Loading SuperAnimal on {device}...")
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
    print("Bodyparts:", bodyparts)
    return pose_runner, detector_runner, bodyparts


def _parse_poses(poses, bodyparts, score_threshold):
    horses = []
    for individual in poses:
        keypoints = []
        for bp_idx, (x, y, score) in enumerate(individual):
            keypoints.append({
                "name":    bodyparts[bp_idx],
                "x":       float(x),
                "y":       float(y),
                "score":   float(score),
                "certain": score >= score_threshold,
            })
        if any(kp["certain"] for kp in keypoints):
            horses.append(keypoints)
    return horses


def run_inference(frame, pose_runner, detector_runner, bodyparts, score_threshold):
    bbox_preds = detector_runner.inference([frame])
    if not bbox_preds:
        return [], []
    pose_preds = pose_runner.inference([(frame, bbox_preds[0])])
    if not pose_preds:
        return [], []
    poses = pose_preds[0].get("bodyparts", None)
    if poses is None:
        return [], []
    raw_boxes = bbox_preds[0].get("bboxes", []) if isinstance(bbox_preds[0], dict) else []
    bboxes = [[float(v) for v in box[:4]] for box in raw_boxes]
    return _parse_poses(poses, bodyparts, score_threshold), bboxes


def draw_horses(frame, horses):
    for horse in horses:
        visible = [kp for kp in horse if any(p in kp["name"].lower() for p in DISPLAY_PARTS)]
        pos = {kp["name"].lower(): (int(kp["x"]), int(kp["y"])) for kp in visible}
        for a, b in SKELETON:
            pa = pos.get(a.lower())
            pb = pos.get(b.lower())
            if pa and pb:
                cv2.line(frame, pa, pb, get_limb_color(a), LINE_THICKNESS, cv2.LINE_AA)
        for kp in visible:
            cx, cy = int(kp["x"]), int(kp["y"])
            color = get_limb_color(kp["name"])
            cv2.circle(frame, (cx, cy), KP_RADIUS, color, -1)
            cv2.putText(frame, kp["name"], (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, 1, cv2.LINE_AA)


VIDEOS = [
    "ref_vids/trot_side.mp4",        # 0
    "ref_vids/canter_slomo.mp4",     # 1
    "ref_vids/canter_graham.mov",    # 2
    "ref_vids/short_trot_Ben.MOV",   # 3
    "ref_vids/walk_highres.mp4",     # 4
]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {i}: {v}" for i, v in enumerate(VIDEOS)),
    )
    parser.add_argument("video", nargs="?", type=int, default=0)
    parser.add_argument("--detector-threshold", type=float, default=DETECTOR_THRESHOLD)
    parser.add_argument("--pose-threshold",     type=float, default=POSE_THRESHOLD)
    parser.add_argument("--interval",           type=int,   default=INFERENCE_INTERVAL)
    parser.add_argument("--max-individuals",    type=int,   default=3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_runner, detector_runner, bodyparts = build_runners(
        args.detector_threshold, args.max_individuals, device
    )

    if args.video < 0 or args.video >= len(VIDEOS):
        print(f"Invalid video index. Choose 0–{len(VIDEOS)-1}.")
        sys.exit(1)
    video_path = VIDEOS[args.video]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video [{args.video}]: {video_path}  {w}x{h}  {fps:.1f}fps")

    transform   = VIDEO_TRANSFORMS.get(args.video, {})
    rotate_code = transform.get("rotate", None)
    disp_scale  = transform.get("scale", 1.0)
    disp_w = int((h if rotate_code is not None else w) * disp_scale)
    disp_h = int((w if rotate_code is not None else h) * disp_scale)

    frame_count      = 0
    horses           = []
    bboxes           = []
    executor         = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inference_future = None
    inference_gray   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if inference_future is not None and inference_future.done():
            try:
                new_horses, new_bboxes = inference_future.result()
                if new_horses:
                    anchor_optical_flow(new_horses, inference_gray)
                    horses = new_horses
                    bboxes = new_bboxes
            except Exception as e:
                print(f"Inference error frame {frame_count}: {e}")
            inference_future = None

        if inference_future is None and frame_count % args.interval == 1:
            inference_gray   = gray.copy()
            inference_future = executor.submit(
                run_inference, frame.copy(), pose_runner, detector_runner,
                bodyparts, args.pose_threshold
            )
            if frame_count == 1:
                print("Waiting for initial detection...")

        horses = track_optical_flow(gray)

        if SHOW_BBOX:
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)

        draw_horses(frame, horses)

        status = f"Frame {frame_count} | {len(horses)} horse(s) | {device.upper()}"
        cv2.putText(frame, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Horse Detection", cv2.resize(frame, (disp_w, disp_h)))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
