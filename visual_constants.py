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

# True  = per-limb colors (magenta/orange/cyan/neon-blue)
# False = flat color (SA → green, custom joints → magenta)
SA_COLOR_TOGGLE    = False
JOINT_COLOR_TOGGLE = True

def get_limb_color(name):
    """SA keypoint color — per-limb when SA_COLOR_TOGGLE, else flat green."""
    if not SA_COLOR_TOGGLE:
        return _GREEN
    n = name.lower()
    if "front_left"  in n: return _MAGENTA
    if "front_right" in n: return _ORANGE
    if "back_left"   in n: return _CYAN
    if "back_right"  in n: return _NEON_BLUE
    return _GREEN

_JOINT_PART_COLORS = {
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

def get_joint_color(name):
    """Custom joint model keypoint color — per-limb when JOINT_COLOR_TOGGLE, else flat magenta."""
    if not JOINT_COLOR_TOGGLE:
        return _MAGENTA
    return _JOINT_PART_COLORS.get(name, _GREEN)

SHOW_BBOX  = True   # Draw horse detection bounding box
BBOX_COLOR = (255, 0, 255)

# --- Inference defaults ---
INFERENCE_INTERVAL   = 3     # Run full inference every N frames; optical flow tracks in between
DETECTOR_THRESHOLD   = 0.5   # Minimum detector confidence to keep a bounding box
POSE_THRESHOLD       = 0.5   # Minimum pose keypoint confidence
JOINTS_THRESHOLD     = 0.01  # Minimum custom joint model confidence
FRAME_PAD_FACTOR     = 0.4   # Fraction of bbox w/h added as padding on each side for joint crop
