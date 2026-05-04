import cv2

# --- Display scale and rotation per video index ---
VIDEO_TRANSFORMS = {
    0: {"scale": 0.6, "flip": 1},                                       # trot_side — horizontal flip
    1: {"scale": 0.6},                                                   # canter_slomo
    2: {"scale": 1.5},                                                   # canter_graham
    3: {"rotate": cv2.ROTATE_90_CLOCKWISE, "max_size": (540, 960)},     # short_trot_Ben — portrait, constrained
    4: {"scale": 0.6},                                                   # walk_highres
    5: {"scale": 0.6},                                                   # austin_trot_left
    6: {"rotate": cv2.ROTATE_90_CLOCKWISE, "max_size": (540, 960)},     # ben_canter — portrait, constrained
    7: {"rotate": cv2.ROTATE_180, "scale": 0.6, "flip": 1},                        # blue_canter — 180°
    8: {"scale": 0.6},                                                   # blue_canter_trot
    9: {"scale": 0.6},                                                   # horse_vid1
   10: {"scale": 0.6},                                                   # horse_vid2
   11: {"scale": 0.2},                                                   # horse_vid3
   12: {"scale": 0.6},                                                   # horse_vid4
   13: {"scale": 0.6, "flip": 1},                                                   # horse_vid5
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
JOINT_COLOR_TOGGLE = False

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

# --- Custom joint model visibility toggles (one per group, covers all four limbs) ---
SHOW_JOINT_HOOVES   = True   # l_f_hoof, r_f_hoof, l_b_hoof, r_b_hoof
SHOW_JOINT_FETLOCKS = True   # l_front_fetlock, r_front_fetlock, l_hind_fetlock, r_hind_fetlock
SHOW_JOINT_KNEES    = False   # l_knee, r_knee, l_hock, r_hock
SHOW_JOINT_SKELETON = False  # skeleton lines between joint model keypoints

# --- SuperAnimal visibility toggles (one per group) ---
SHOW_SA_BODY     = False   # nose, throat_base, back_base, back_end, tail_base
SHOW_SA_THAIS    = False   # *_thai (upper leg)
SHOW_SA_KNEES    = False   # *_knee
SHOW_SA_PAWS     = True   # *_paw
SHOW_SA_FETLOCKS = False  # fetlock points from custom model, inserted between knee and paw
SHOW_SA_SKELETON = False  # skeleton lines between SA keypoints

SHOW_BBOX  = True   # Draw horse detection bounding box
BBOX_COLOR = (255, 0, 255)

# --- Inference defaults ---
INFERENCE_INTERVAL   = 3     # Run full inference every N frames; optical flow tracks in between
DETECTOR_THRESHOLD   = 0.5   # Minimum detector confidence to keep a bounding box
POSE_THRESHOLD       = 0.6   # Minimum pose keypoint confidence
JOINTS_THRESHOLD     = 0.4  # Minimum custom joint model confidence
FRAME_PAD_FACTOR     = 0.6   # Fraction of bbox w/h added as padding on each side for joint crop
