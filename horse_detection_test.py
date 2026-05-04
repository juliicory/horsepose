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
    INFERENCE_INTERVAL, DETECTOR_THRESHOLD, POSE_THRESHOLD,
    SHOW_BBOX, BBOX_COLOR,
    SHOW_SA_BODY, SHOW_SA_THAIS, SHOW_SA_KNEES, SHOW_SA_PAWS,
    SHOW_SA_FETLOCKS, SHOW_SA_SKELETON,
    SA_COLOR_TOGGLE,
    _MAGENTA, _ORANGE, _CYAN, _NEON_BLUE, _GREEN,
)

def _limb_color(name, toggle):
    if not toggle:
        return _GREEN
    n = name.lower()
    if "front_left"  in n: return _MAGENTA
    if "front_right" in n: return _ORANGE
    if "back_left"   in n: return _CYAN
    if "back_right"  in n: return _NEON_BLUE
    return _GREEN


def _sa_visible(name, state=None):
    n = name.lower()
    if state is not None:
        if "thai" in n: return state.show_sa_thais
        if "knee" in n: return state.show_sa_knees
        if "paw"  in n: return state.show_sa_paws
        return state.show_sa_body
    if "thai" in n: return SHOW_SA_THAIS
    if "knee" in n: return SHOW_SA_KNEES
    if "paw"  in n: return SHOW_SA_PAWS
    return SHOW_SA_BODY  # nose, throat_base, back_base, back_end, tail_base

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

# Skeleton with fetlock points inserted between knee and paw
SKELETON_WITH_FETLOCKS = [
    ("nose",                   "throat_base"),
    ("throat_base",            "back_base"),
    ("back_base",              "back_end"),
    ("back_end",               "tail_base"),
    ("back_base",              "front_left_thai"),
    ("front_left_thai",        "front_left_knee"),
    ("front_left_knee",        "front_left_fetlock"),
    ("front_left_fetlock",     "front_left_paw"),
    ("back_base",              "front_right_thai"),
    ("front_right_thai",       "front_right_knee"),
    ("front_right_knee",       "front_right_fetlock"),
    ("front_right_fetlock",    "front_right_paw"),
    ("back_end",               "back_left_thai"),
    ("back_left_thai",         "back_left_knee"),
    ("back_left_knee",         "back_left_fetlock"),
    ("back_left_fetlock",      "back_left_paw"),
    ("back_end",               "back_right_thai"),
    ("back_right_thai",        "back_right_knee"),
    ("back_right_knee",        "back_right_fetlock"),
    ("back_right_fetlock",     "back_right_paw"),
]

# Maps custom joint model fetlock names → artificial SA-style pos-dict keys
_FETLOCK_SA_KEY = {
    "l_front_fetlock": "front_left_fetlock",
    "r_front_fetlock": "front_right_fetlock",
    "l_hind_fetlock":  "back_left_fetlock",
    "r_hind_fetlock":  "back_right_fetlock",
}

# SA leg chains top→bottom: knee → (fetlock) → paw
_SA_LEG_CHAINS = [
    ("front_left_knee",  "front_left_fetlock",  "front_left_paw"),
    ("front_right_knee", "front_right_fetlock",  "front_right_paw"),
    ("back_left_knee",   "back_left_fetlock",    "back_left_paw"),
    ("back_right_knee",  "back_right_fetlock",   "back_right_paw"),
]


def _apply_vertical_constraint(pos, *keys):
    """For 3-key chains (knee, fetlock, paw) where all three are present:
    clamp the fetlock y to the 1/4–1/3 span between knee and paw.
    Falls back to simple monotone ordering otherwise."""
    if len(keys) == 3:
        top_k, mid_k, bot_k = keys
        if top_k in pos and bot_k in pos and mid_k in pos:
            top_y = pos[top_k][1]
            bot_y = pos[bot_k][1]
            span  = bot_y - top_y
            if span > 0:
                lo = top_y + span * 0.75
                hi = top_y + span / 5.0
                mx, my = pos[mid_k]
                pos[mid_k] = (mx, int(round(max(lo, min(hi, float(my))))))
                return
            # span <= 0: fall through to simple ordering

    prev_y = None
    for key in keys:
        if key not in pos:
            continue
        x, y = pos[key]
        if prev_y is not None and y < prev_y:
            pos[key] = (x, prev_y)
        else:
            prev_y = y

# --- OPTICAL FLOW ---
def reset_optical_flow():
    global _prev_gray, _of_points, _of_meta
    _prev_gray = None
    _of_points = None
    _of_meta   = []


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


def draw_horses(frame, horses, state=None, joint_horses=None):
    _color        = state.sa_color_toggle  if state else SA_COLOR_TOGGLE
    _skel         = state.show_sa_skeleton if state else SHOW_SA_SKELETON
    _show_fetlock = (state.show_sa_fetlocks if state else SHOW_SA_FETLOCKS)
    _kp_r         = state.kp_radius        if state else KP_RADIUS
    _txt          = state.text_scale       if state else TEXT_SCALE
    _thick        = state.line_thickness   if state else LINE_THICKNESS
    skeleton      = SKELETON_WITH_FETLOCKS if _show_fetlock else SKELETON

    for horse_idx, horse in enumerate(horses):
        visible = [kp for kp in horse if _sa_visible(kp["name"], state)]
        pos = {kp["name"].lower(): (int(kp["x"]), int(kp["y"])) for kp in visible}

        # Inject fetlock positions from the custom joint model into the SA pos dict
        if _show_fetlock and joint_horses and horse_idx < len(joint_horses):
            for kp in joint_horses[horse_idx]:
                if not kp.get("certain", True):
                    continue
                sa_key = _FETLOCK_SA_KEY.get(kp["name"])
                if sa_key:
                    pos[sa_key] = (int(kp["x"]), int(kp["y"]))

        # Enforce knee ≥ fetlock ≥ paw in vertical position
        for chain in _SA_LEG_CHAINS:
            _apply_vertical_constraint(pos, *chain)

        # Draw injected fetlock dots after constraint
        if _show_fetlock and joint_horses and horse_idx < len(joint_horses):
            for kp in joint_horses[horse_idx]:
                sa_key = _FETLOCK_SA_KEY.get(kp["name"])
                if sa_key and sa_key in pos:
                    cx, cy = pos[sa_key]
                    color  = _limb_color(sa_key, _color)
                    cv2.circle(frame, (cx, cy), _kp_r, color, -1)
                    cv2.putText(frame, kp["name"], (cx + 6, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, _txt, color, 1, cv2.LINE_AA)

        if _skel:
            for a, b in skeleton:
                pa = pos.get(a.lower())
                pb = pos.get(b.lower())
                if pa and pb:
                    cv2.line(frame, pa, pb, _limb_color(a, _color), _thick, cv2.LINE_AA)
        for kp in visible:
            cx, cy = int(kp["x"]), int(kp["y"])
            color = _limb_color(kp["name"], _color)
            cv2.circle(frame, (cx, cy), _kp_r, color, -1)
            cv2.putText(frame, kp["name"], (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _txt, color, 1, cv2.LINE_AA)


VIDEOS = [
    "ref_vids/trot_side.mp4",                                               # 0
    "ref_vids/canter_slomo.mp4",                                            # 1
    "ref_vids/canter_graham.mov",                                           # 2
    "ref_vids/short_trot_Ben.MOV",                                          # 3
    "ref_vids/walk_highres.mp4",                                            # 4
    "horse_joints-julic-2026-04-29/videos/austin_trot_left.MOV",           # 5
    "horse_joints-julic-2026-04-29/videos/ben_canter.MOV",                 # 6
    "horse_joints-julic-2026-04-29/videos/blue_canter.mov",                # 7
    "horse_joints-julic-2026-04-29/videos/blue_canter_trot.mov",           # 8
    "horse_joints-julic-2026-04-29/videos/horse_vid1.mp4",                 # 9
    "horse_joints-julic-2026-04-29/videos/horse_vid2.mp4",                 # 10
    "horse_joints-julic-2026-04-29/videos/horse_vid3.mp4",                 # 11
    "horse_joints-julic-2026-04-29/videos/horse_vid4.mp4",                 # 12
    "horse_joints-julic-2026-04-29/videos/horse_vid5.mp4",                 # 13
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
