"""
ResNet joint detector test — SA body pose + custom joint model (hoof/fetlock/knee/hock).
Cyan overlay = custom joint model. Colored overlay = SuperAnimal. Press Q to quit.

Usage:
    python combined_detection.py
    python combined_detection.py 3
    python combined_detection.py 4 --joints-threshold 0.15
"""

import sys
import argparse
import concurrent.futures
import cv2
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis import get_pose_inference_runner

from horse_detection_test import (
    _horses_to_points, _points_to_horses,
    anchor_optical_flow, track_optical_flow,
    _parse_poses, draw_horses,
    build_runners as build_sa_runners,
    VIDEOS, LK_PARAMS,
)
from visual_constants import (
    VIDEO_TRANSFORMS, KP_RADIUS_JOINT, TEXT_SCALE, LINE_THICKNESS,
    INFERENCE_INTERVAL, DETECTOR_THRESHOLD, POSE_THRESHOLD, JOINTS_THRESHOLD,
    JOINT_PART_COLORS, SHOW_BBOX, BBOX_COLOR,
)

JOINTS_CONFIG   = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\dlc-models-pytorch\iteration-0\horse_jointsApr29-trainset95shuffle9\train\pytorch_config.yaml"
JOINTS_SNAPSHOT = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\dlc-models-pytorch\iteration-0\horse_jointsApr29-trainset95shuffle9\train\snapshot-best-040.pt"

SHOW_SUPERANIMAL = False
SHOW_JOINTS      = True

JOINT_SKELETON = [
    ("l_f_hoof", "l_front_fetlock"), ("l_front_fetlock", "l_knee"),
    ("r_f_hoof", "r_front_fetlock"), ("r_front_fetlock", "r_knee"),
    ("l_b_hoof", "l_hind_fetlock"),  ("l_hind_fetlock",  "l_hock"),
    ("r_b_hoof", "r_hind_fetlock"),  ("r_hind_fetlock",  "r_hock"),
]

# Joint optical flow state (separate from SA state in horse_detection_test)
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


def run_inference(frame, pose_runner, detector_runner, bodyparts,
                  joints_runner, joints_bodyparts, score_threshold, joints_threshold):
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
    bboxes = [[float(v) for v in box[:4]] for box in raw_boxes]
    horses = _parse_poses(poses, bodyparts, score_threshold)

    JOINT_INPUT_SIZE = 448
    joint_horses = []
    if not SHOW_JOINTS or joints_runner is None:
        return horses, bboxes, joint_horses
    fh, fw = frame.shape[:2]
    for box in raw_boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop_w, crop_h = x2 - x1, y2 - y1
        crop = cv2.resize(frame[y1:y2, x1:x2], (JOINT_INPUT_SIZE, JOINT_INPUT_SIZE))
        scale_x = crop_w / JOINT_INPUT_SIZE
        scale_y = crop_h / JOINT_INPUT_SIZE
        try:
            preds = joints_runner.inference([crop])
            if not preds:
                continue
            joint_poses = preds[0].get("bodyparts", None)
            if joint_poses is None or len(joint_poses) == 0:
                continue
            keypoints = []
            for bp_idx, (px, py, score) in enumerate(joint_poses[0]):
                keypoints.append({
                    "name":    joints_bodyparts[bp_idx],
                    "x":       float(x1 + px * scale_x),
                    "y":       float(y1 + py * scale_y),
                    "score":   float(score),
                    "certain": score >= joints_threshold,
                })
            if any(kp["certain"] for kp in keypoints):
                joint_horses.append(keypoints)
        except Exception as e:
            print(f"Joint inference error: {e}")

    return horses, bboxes, joint_horses


_DEFAULT_JOINT_COLOR = (0, 255, 0)

def draw_joints(frame, joint_horses):
    for horse in joint_horses:
        pos = {kp["name"]: (int(kp["x"]), int(kp["y"]))
               for kp in horse if kp.get("certain", True)}
        for a, b in JOINT_SKELETON:
            if a in pos and b in pos:
                color = JOINT_PART_COLORS.get(a, _DEFAULT_JOINT_COLOR)
                cv2.line(frame, pos[a], pos[b], color, LINE_THICKNESS, cv2.LINE_AA)
        for name, (cx, cy) in pos.items():
            color = JOINT_PART_COLORS.get(name, _DEFAULT_JOINT_COLOR)
            cv2.circle(frame, (cx, cy), KP_RADIUS_JOINT, color, -1)
            cv2.putText(frame, name, (cx + 6, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, 1, cv2.LINE_AA)


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
    parser.add_argument("--joints-threshold",   type=float, default=JOINTS_THRESHOLD)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_runner, detector_runner, bodyparts = build_sa_runners(
        args.detector_threshold, args.max_individuals, device
    )
    joints_runner, joints_bodyparts = build_joint_runner(args.max_individuals, device) \
        if SHOW_JOINTS else (None, [])

    if args.video < 0 or args.video >= len(VIDEOS):
        print(f"Invalid video index. Choose 0–{len(VIDEOS)-1}.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEOS[args.video])
    if not cap.isOpened():
        print(f"Could not open video: {VIDEOS[args.video]}")
        sys.exit(1)

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    transform   = VIDEO_TRANSFORMS.get(args.video, {})
    rotate_code = transform.get("rotate", None)
    disp_scale  = transform.get("scale", 1.0)
    disp_w = int((h if rotate_code is not None else w) * disp_scale)
    disp_h = int((w if rotate_code is not None else h) * disp_scale)

    frame_count  = 0
    horses       = []
    bboxes       = []
    joint_horses = []
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
                new_horses, new_bboxes, new_joint_horses = inference_future.result()
                if new_horses:
                    anchor_optical_flow(new_horses, inference_gray)
                    horses = new_horses
                    bboxes = new_bboxes
                if new_joint_horses:
                    anchor_joint_optical_flow(new_joint_horses, inference_gray)
                    joint_horses = new_joint_horses
            except Exception as e:
                print(f"Inference error frame {frame_count}: {e}")
            inference_future = None

        if inference_future is None and frame_count % args.interval == 1:
            inference_gray   = gray.copy()
            inference_future = executor.submit(
                run_inference, frame.copy(), pose_runner, detector_runner,
                bodyparts, joints_runner, joints_bodyparts,
                args.pose_threshold, args.joints_threshold
            )
            if frame_count == 1:
                print("Waiting for initial detection...")

        horses = track_optical_flow(gray)
        if SHOW_JOINTS:
            joint_horses = track_joint_optical_flow(gray)

        if SHOW_BBOX:
            for x1, y1, x2, y2 in bboxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)

        if SHOW_SUPERANIMAL:
            draw_horses(frame, horses)
        if SHOW_JOINTS:
            draw_joints(frame, joint_horses)

        status = f"Frame {frame_count} | {len(horses)} horse(s) | {device.upper()}"
        cv2.putText(frame, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Joint Detector", cv2.resize(frame, (disp_w, disp_h)))
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    executor.shutdown(wait=False)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
