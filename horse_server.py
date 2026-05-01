import asyncio
import concurrent.futures
import websockets
import json
import cv2
import numpy as np
import torch
import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
from deeplabcut.pose_estimation_pytorch.apis import (
    get_pose_inference_runner,
    get_detector_inference_runner,
)

# --- CONFIGURATION ---
# Swap VIDEO_SOURCE for a different camera or virtual camera without changing anything else.
# 0 = first webcam (Windows). For Quest passthrough via virtual camera, point this to that device index.
VIDEO_SOURCE      = 0
HOST              = "localhost"
PORT              = 8765
SCORE_THRESHOLD   = 0.5
MAX_INDIVIDUALS   = 3
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# Full DLC inference fires every N frames; optical flow tracks keypoints in between.
# Lower = more accurate, slower. Higher = faster, more drift.
KEYFRAME_INTERVAL = 1
# Substrings to match for preview labels — head landmarks and hooves/paws only.
# All keypoints are still broadcast to TD.
DISPLAY_PARTS     = ("nose", "hoof", "paw")

# --- LOAD MODEL ---
print(f"Loading model on {DEVICE}...")
_config = modelzoo.load_super_animal_config(
    super_animal="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
)
_config["metadata"]["individuals"] = [f"animal{i}" for i in range(MAX_INDIVIDUALS)]
if _config.get("detector") is not None:
    _config["detector"]["model"]["box_score_thresh"] = 0.6

BODYPARTS = _config["metadata"]["bodyparts"]

pose_runner = get_pose_inference_runner(
    model_config=_config,
    snapshot_path=modelzoo.get_super_animal_snapshot_path(
        dataset="superanimal_quadruped", model_name="hrnet_w32"
    ),
    max_individuals=MAX_INDIVIDUALS,
    device=DEVICE,
)
detector_runner = get_detector_inference_runner(
    model_config=_config,
    snapshot_path=modelzoo.get_super_animal_snapshot_path(
        dataset="superanimal_quadruped", model_name="fasterrcnn_resnet50_fpn_v2"
    ),
    max_individuals=MAX_INDIVIDUALS,
    device=DEVICE,
)
print("Model ready.")

# --- INFERENCE ---
def run_inference(frame):
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
                "name":    BODYPARTS[bp_idx],
                "x":       round(float(x), 2),
                "y":       round(float(y), 2),
                "score":   round(float(score), 3),
                "certain": score >= SCORE_THRESHOLD,
            })
        if any(kp["certain"] for kp in keypoints):
            horses.append(keypoints)
    return horses

# --- OPTICAL FLOW ---
# Mirrors how TD skeleton tracking interpolates between sensor ticks:
# DLC inference = slow ground truth (like a Kinect frame), optical flow = cheap
# per-frame update that keeps keypoints moving smoothly between those ticks.

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

_prev_gray = None   # grayscale frame used as optical flow reference
_of_points = None   # np.float32 [total_kps, 1, 2] — all individuals flattened
_of_meta   = []     # [(animal_idx, name, score), ...] parallel to _of_points rows


def _horses_to_points(horses):
    """Flatten all keypoints into one numpy array for LK."""
    points, meta = [], []
    for animal_idx, horse in enumerate(horses):
        for kp in horse:
            points.append([[kp["x"], kp["y"]]])
            meta.append((animal_idx, kp["name"], kp["score"]))
    if points:
        return np.array(points, dtype=np.float32), meta
    return None, []


def _points_to_horses(points, meta):
    """Reconstruct horses list from tracked point positions."""
    if points is None or not meta:
        return []
    n_animals = max(m[0] for m in meta) + 1
    horses = [[] for _ in range(n_animals)]
    for pt, (animal_idx, name, score) in zip(points, meta):
        x, y = pt[0]
        horses[animal_idx].append({
            "name":  name,
            "x":     round(float(x), 2),
            "y":     round(float(y), 2),
            "score": round(float(score), 3),
        })
    return [h for h in horses if h]


def anchor_optical_flow(horses, gray):
    """Reset optical flow to a fresh inference result, falling back to last
    tracked positions for uncertain keypoints."""
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
    """Shift keypoints from prev frame to current frame using Lucas-Kanade."""
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
                _of_points[i] = pt  # tracked ok — update position
            # else: keep last known position (point was lost)

    return _points_to_horses(_of_points, _of_meta)


# --- WEBSOCKET ---
# JSON schema: { "frame": int, "horses": [ [ { "name", "x", "y", "score" }, ... ], ... ] }
connected_clients = set()

async def broadcast(message):
    if connected_clients:
        await asyncio.gather(*[c.send(message) for c in connected_clients])

async def handler(websocket):
    connected_clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print("Client disconnected.")

# --- MAIN LOOP ---
async def capture_and_infer():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video source: {VIDEO_SOURCE}  resolution: {w}x{h}  fps: {fps}")
    print(f"WebSocket at ws://{HOST}:{PORT}")
    print(f"Inference every {KEYFRAME_INTERVAL} frames, optical flow in between.")

    frame_count = 0
    horses = []
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    inference_future = None
    inference_gray = None   # grayscale of the frame sent to inference

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(VIDEO_SOURCE, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("Webcam lost.")
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Check if last inference finished; anchor optical flow to it ---
        if inference_future is not None and inference_future.done():
            try:
                new_horses = inference_future.result()
                if new_horses:
                    # Anchor using the gray from the frame that was inferred,
                    # so optical flow starts from the right reference point.
                    anchor_optical_flow(new_horses, inference_gray)
                    horses = _points_to_horses(_of_points, _of_meta)
            except Exception as e:
                print(f"Inference error frame {frame_count}: {e}")
            inference_future = None

        # --- Fire new inference every KEYFRAME_INTERVAL frames ---
        if inference_future is None and frame_count % KEYFRAME_INTERVAL == 0:
            inference_gray = gray.copy()
            inference_future = loop.run_in_executor(executor, run_inference, frame.copy())

        # --- Optical flow: track keypoints every frame (fast, CPU, ~1ms) ---
        horses = track_optical_flow(gray)

        if frame_count % 30 == 0:
            print(f"Frame {frame_count} — {len(horses)} horse(s) tracked")

        # --- Preview (head + hooves only) ---
        for horse in horses:
            for kp in horse:
                if not any(part in kp["name"] for part in DISPLAY_PARTS):
                    continue
                cx, cy = int(kp["x"]), int(kp["y"])
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, kp["name"], (cx + 6, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Horse Pose Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Preview closed.")
            break

        await broadcast(json.dumps({"frame": frame_count, "horses": horses}))
        await asyncio.sleep(0)

    cap.release()
    cv2.destroyAllWindows()

# --- ENTRY POINT ---
async def main():
    async with websockets.serve(handler, HOST, PORT):
        await capture_and_infer()

asyncio.run(main())
