"""
SA body pose + custom joint model (hoof/fetlock/knee/hock).
A tkinter control panel opens alongside the video window.
Press Q (in the video window) or close the control panel to quit.

Usage:
    python combined_detection.py
    python combined_detection.py 3
    python combined_detection.py 4 --joints-threshold 0.15
"""

import sys
import argparse
import threading
import concurrent.futures
import tkinter as tk
import cv2
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis import get_pose_inference_runner

from horse_detection_test import (
    _horses_to_points, _points_to_horses,
    anchor_optical_flow, track_optical_flow, reset_optical_flow,
    _parse_poses, draw_horses, _apply_vertical_constraint,
    build_runners as build_sa_runners,
    VIDEOS, LK_PARAMS,
)
from visual_constants import (
    VIDEO_TRANSFORMS, BBOX_COLOR,
    INFERENCE_INTERVAL, DETECTOR_THRESHOLD, POSE_THRESHOLD, JOINTS_THRESHOLD,
)
import visual_constants as _vc
from combined_detector_ui import UIState, ControlPanel

JOINTS_CONFIG   = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\dlc-models-pytorch\iteration-0\horse_jointsApr29-trainset95shuffle9\train\pytorch_config.yaml"
JOINTS_SNAPSHOT = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\dlc-models-pytorch\iteration-0\horse_jointsApr29-trainset95shuffle9\train\snapshot-best-190.pt"

JOINT_SKELETON = [
    ("l_f_hoof", "l_front_fetlock"), ("l_front_fetlock", "l_knee"),
    ("r_f_hoof", "r_front_fetlock"), ("r_front_fetlock", "r_knee"),
    ("l_b_hoof", "l_hind_fetlock"),  ("l_hind_fetlock",  "l_hock"),
    ("r_b_hoof", "r_hind_fetlock"),  ("r_hind_fetlock",  "r_hock"),
]

# Custom model leg chains top→bottom: knee/hock → fetlock → hoof
_JOINT_LEG_CHAINS = [
    ("l_knee", "l_front_fetlock", "l_f_hoof"),
    ("r_knee", "r_front_fetlock", "r_f_hoof"),
    ("l_hock", "l_hind_fetlock",  "l_b_hoof"),
    ("r_hock", "r_hind_fetlock",  "r_b_hoof"),
]

_j_prev_gray = None
_j_of_points = None
_j_of_meta   = []


# ── Optical flow (joint model) ────────────────────────────────────────────────

def _reset_joint_of():
    global _j_prev_gray, _j_of_points, _j_of_meta
    _j_prev_gray = None
    _j_of_points = None
    _j_of_meta   = []


def anchor_joint_optical_flow(joint_horses, gray):
    global _j_prev_gray, _j_of_points, _j_of_meta
    _j_prev_gray = gray
    prev_lookup = {}
    if _j_of_points is not None:
        for pt, (animal_idx, name, _) in zip(_j_of_points, _j_of_meta):
            prev_lookup[(animal_idx, name)] = pt[0]
    merged = []
    for animal_idx, horse in enumerate(joint_horses):
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
    _j_of_points, _j_of_meta = _horses_to_points(merged)


def track_joint_optical_flow(gray):
    global _j_prev_gray, _j_of_points
    if _j_prev_gray is None or _j_of_points is None or len(_j_of_points) == 0:
        _j_prev_gray = gray
        return _points_to_horses(_j_of_points, _j_of_meta)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        _j_prev_gray, gray, _j_of_points, None, **LK_PARAMS
    )
    _j_prev_gray = gray
    if new_pts is not None and status is not None:
        for i, (ok, pt) in enumerate(zip(status, new_pts)):
            if ok[0]:
                _j_of_points[i] = pt
    return _points_to_horses(_j_of_points, _j_of_meta)


# ── Model loader ──────────────────────────────────────────────────────────────

def build_joint_runner(max_individuals, device):
    print("Loading joint model...")
    with open(JOINTS_CONFIG) as f:
        joints_config = yaml.safe_load(f)
    joints_config["metadata"]["individuals"] = [f"animal{i}" for i in range(max_individuals)]
    joints_bodyparts = joints_config["metadata"]["bodyparts"]
    joints_runner = get_pose_inference_runner(
        model_config=joints_config,
        snapshot_path=JOINTS_SNAPSHOT,
        max_individuals=max_individuals,
        device=device,
    )
    return joints_runner, joints_bodyparts


# ── Inference ─────────────────────────────────────────────────────────────────

def _pad_box(box, frame_h, frame_w, pad_factor):
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    pw = (x2 - x1) * pad_factor
    ph = (y2 - y1) * pad_factor
    padded = list(box)
    padded[0] = max(0.0,            x1 - pw)
    padded[1] = max(0.0,            y1 - ph)
    padded[2] = min(float(frame_w), x2 + pw)
    padded[3] = min(float(frame_h), y2 + ph)
    return padded


def run_inference(frame, pose_runner, detector_runner, bodyparts,
                  joints_runner, joints_bodyparts,
                  score_threshold, joints_threshold, show_joints, frame_pad_factor):
    bbox_preds = detector_runner.inference([frame])
    if not bbox_preds:
        return [], [], []
    pose_preds = pose_runner.inference([(frame, bbox_preds[0])])
    if not pose_preds:
        return [], [], []
    poses = pose_preds[0].get("bodyparts", None)
    if poses is None:
        return [], [], []
    raw_boxes = bbox_preds[0].get("bboxes", []) if isinstance(bbox_preds[0], dict) else []
    bboxes    = [[float(v) for v in box[:4]] for box in raw_boxes]
    horses    = _parse_poses(poses, bodyparts, score_threshold)

    joint_horses = []
    if not show_joints or joints_runner is None:
        return horses, bboxes, joint_horses

    fh, fw = frame.shape[:2]
    for box in raw_boxes:
        padded_box = _pad_box(box, fh, fw, frame_pad_factor)
        try:
            preds = joints_runner.inference([(frame, {"bboxes": [padded_box]})])
            if not preds:
                continue
            joint_poses = preds[0].get("bodyparts", None)
            if joint_poses is None:
                continue
            for individual in joint_poses:
                keypoints = []
                for bp_idx, kp_data in enumerate(individual):
                    if len(kp_data) >= 3:
                        px, py, score = float(kp_data[0]), float(kp_data[1]), float(kp_data[2])
                    else:
                        px, py, score = float(kp_data[0]), float(kp_data[1]), 1.0
                    keypoints.append({
                        "name":    joints_bodyparts[bp_idx],
                        "x":       px,
                        "y":       py,
                        "score":   score,
                        "certain": score >= joints_threshold,
                    })
                if any(kp["certain"] for kp in keypoints):
                    joint_horses.append(keypoints)
        except Exception as e:
            print(f"Joint inference error: {e}")

    return horses, bboxes, joint_horses


# ── Drawing ───────────────────────────────────────────────────────────────────

def _joint_color(name, state):
    if not state.joint_color_toggle:
        return _vc._MAGENTA
    return _vc._JOINT_PART_COLORS.get(name, _vc._GREEN)


def _joint_visible(name, state):
    if "fetlock" in name: return state.show_joint_fetlocks
    if "hoof"    in name: return state.show_joint_hooves
    return state.show_joint_knees


def draw_joints(frame, joint_horses, state):
    for horse in joint_horses:
        pos = {kp["name"]: (int(kp["x"]), int(kp["y"]))
               for kp in horse
               if kp.get("certain", True) and _joint_visible(kp["name"], state)}
        for chain in _JOINT_LEG_CHAINS:
            _apply_vertical_constraint(pos, *chain)
        if state.show_joint_skeleton:
            for a, b in JOINT_SKELETON:
                if a in pos and b in pos:
                    cv2.line(frame, pos[a], pos[b], _joint_color(a, state),
                             state.line_thickness, cv2.LINE_AA)
        for name, (cx, cy) in pos.items():
            color = _joint_color(name, state)
            cv2.circle(frame, (cx, cy), state.kp_radius_joint, color, -1)
            cv2.putText(frame, name, (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, state.text_scale, color, 1, cv2.LINE_AA)


# ── Video loop (background thread) ───────────────────────────────────────────

def _video_loop(stop_event, state,
                pose_runner, detector_runner, bodyparts,
                joints_runner, joints_bodyparts):

    current_video_idx = -1
    cap         = None
    rotate_code = None
    flip_code   = None
    disp_w = disp_h = 640
    fps = 30

    frame_count      = 0
    horses           = []
    bboxes           = []
    joint_horses     = []
    executor         = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inference_future = None
    inference_gray   = None

    while not stop_event.is_set():

        # ── Video switching ──────────────────────────────────────────────
        if state.video_index != current_video_idx:
            if cap is not None:
                cap.release()
            video_path = VIDEOS[state.video_index]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open: {video_path}")
                state.video_index = current_video_idx  # revert
                continue
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            transform   = VIDEO_TRANSFORMS.get(state.video_index, {})
            rotate_code = transform.get("rotate", None)
            flip_code   = transform.get("flip",   None)
            swaps_dims  = rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_w     = h if swaps_dims else w
            frame_h     = w if swaps_dims else h
            max_size    = transform.get("max_size", None)
            if max_size is not None:
                scale  = min(max_size[0] / frame_w, max_size[1] / frame_h)
            else:
                scale  = transform.get("scale", 1.0)
            disp_w = int(frame_w * scale)
            disp_h = int(frame_h * scale)
            current_video_idx = state.video_index
            frame_count  = 0
            horses       = []
            bboxes       = []
            joint_horses = []
            inference_future = None
            reset_optical_flow()
            _reset_joint_of()
            print(f"Loaded video {state.video_index}: {video_path}")

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)
        if flip_code is not None:
            frame = cv2.flip(frame, flip_code)

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Inference results ────────────────────────────────────────────
        if inference_future is not None and inference_future.done():
            try:
                new_horses, new_bboxes, new_joint_horses = inference_future.result()
                if new_horses:
                    anchor_optical_flow(new_horses, inference_gray)
                    horses       = new_horses
                    bboxes       = new_bboxes
                if new_joint_horses:
                    anchor_joint_optical_flow(new_joint_horses, inference_gray)
                    joint_horses = new_joint_horses
            except Exception as e:
                print(f"Inference error frame {frame_count}: {e}")
            inference_future = None

        if inference_future is None and frame_count % state.inference_interval == 1:
            inference_gray   = gray.copy()
            inference_future = executor.submit(
                run_inference, frame.copy(),
                pose_runner, detector_runner, bodyparts,
                joints_runner, joints_bodyparts,
                state.pose_threshold, state.joints_threshold,
                state.show_joints, state.frame_pad_factor,
            )
            if frame_count == 1:
                print("Waiting for initial detection...")

        horses = track_optical_flow(gray)
        if state.show_joints:
            joint_horses = track_joint_optical_flow(gray)

        # ── Drawing ──────────────────────────────────────────────────────
        if state.show_bbox:
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)

        if state.show_superanimal:
            draw_horses(frame, horses, state,
                        joint_horses if state.show_sa_fetlocks else None)
        if state.show_joints:
            draw_joints(frame, joint_horses, state)

        cv2.putText(frame, f"Frame {frame_count} | {len(horses)} horse(s)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if state.debug_joint_crop and bboxes:
            fh2, fw2 = frame.shape[:2]
            pb  = _pad_box(bboxes[0], fh2, fw2, state.frame_pad_factor)
            x1, y1, x2, y2 = [int(v) for v in pb[:4]]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imshow("Joint crop", cv2.resize(crop, (448, 448)))

        cv2.imshow("Joint Detector", cv2.resize(frame, (disp_w, disp_h)))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            stop_event.set()
            break

    executor.shutdown(wait=False)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {i}: {v}" for i, v in enumerate(VIDEOS)),
    )
    parser.add_argument("video",                nargs="?", type=int,   default=0)
    parser.add_argument("--detector-threshold", type=float, default=DETECTOR_THRESHOLD)
    parser.add_argument("--pose-threshold",     type=float, default=POSE_THRESHOLD)
    parser.add_argument("--interval",           type=int,   default=INFERENCE_INTERVAL)
    parser.add_argument("--max-individuals",    type=int,   default=3)
    parser.add_argument("--joints-threshold",   type=float, default=JOINTS_THRESHOLD)
    args = parser.parse_args()

    if args.video < 0 or args.video >= len(VIDEOS):
        print(f"Invalid video index. Choose 0–{len(VIDEOS)-1}.")
        sys.exit(1)

    state  = UIState(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pose_runner, detector_runner, bodyparts = build_sa_runners(
        args.detector_threshold, args.max_individuals, device
    )
    joints_runner, joints_bodyparts = build_joint_runner(args.max_individuals, device)

    stop_event   = threading.Event()
    video_thread = threading.Thread(
        target=_video_loop,
        args=(stop_event, state,
              pose_runner, detector_runner, bodyparts,
              joints_runner, joints_bodyparts),
        daemon=True,
    )
    video_thread.start()

    root = tk.Tk()
    ControlPanel(root, state, VIDEOS)
    root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
    root.mainloop()
    stop_event.set()


if __name__ == "__main__":
    main()
