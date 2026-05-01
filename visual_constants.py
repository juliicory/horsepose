import cv2

# --- Display scale and rotation per video index ---
VIDEO_TRANSFORMS = {
    0: {"scale": 0.6},
    1: {"scale": 0.6},
    2: {"scale": 1.5},
    3: {"rotate": cv2.ROTATE_90_CLOCKWISE, "scale": 0.6},
    4: {"scale": 0.6},
}

# --- Keypoint marker sizes ---
KP_RADIUS       = 5    # SuperAnimal keypoint circle radius (px)
KP_RADIUS_JOINT = 6    # Custom joint model circle radius (px)

ANIMAL_COLORS = [
    (0,   255,   0),
    (0,   128, 255),
    (255,  64,  64),
]

# --- Label text ---
TEXT_SCALE     = 0.5   # Font scale for keypoint labels
LINE_THICKNESS = 2     # Skeleton line thickness

# --- Joint model keypoint colors (BGR) ---
_MAGENTA   = (255,   0, 255)   # front left
_ORANGE    = (  0, 120, 255)   # front right
_CYAN      = (255, 255,   0)   # back left
_NEON_BLUE = (255,  10,   0)   # back right

_GREEN = (0, 255, 0)

def get_limb_color(name):
    """Return limb color for a keypoint name based on front/back left/right substrings."""
    n = name.lower()
    if "front_left"  in n: return _MAGENTA
    if "front_right" in n: return _ORANGE
    if "back_left"   in n: return _CYAN
    if "back_right"  in n: return _NEON_BLUE
    return _GREEN

JOINT_PART_COLORS = {
    "l_f_hoof":        _MAGENTA,
    "l_front_fetlock": _MAGENTA,
    "l_knee":          _MAGENTA,
    "r_f_hoof":        _ORANGE,
    "r_front_fetlock": _ORANGE,
    "r_knee":          _ORANGE,
    "l_b_hoof":        _CYAN,
    "l_hind_fetlock":  _CYAN,
    "l_hock":          _CYAN,
    "r_b_hoof":        _NEON_BLUE,
    "r_hind_fetlock":  _NEON_BLUE,
    "r_hock":          _NEON_BLUE,
}

SHOW_BBOX  = False   # Draw horse detection bounding box
BBOX_COLOR = (255, 0, 255)

# --- Inference defaults ---
INFERENCE_INTERVAL   = 4     # Run full inference every N frames; optical flow tracks in between
DETECTOR_THRESHOLD   = 0.7   # Minimum detector confidence to keep a bounding box
POSE_THRESHOLD       = 0.7   # Minimum pose keypoint confidence
JOINTS_THRESHOLD     = 0.2   # Minimum custom joint model confidence
