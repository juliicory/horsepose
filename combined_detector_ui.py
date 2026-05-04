"""
Tkinter control panel for combined_detection.py.
UIState holds all mutable runtime state (defaults loaded from visual_constants).
ControlPanel builds a scrollable window with collapsible sections that mirror
the sections defined in visual_constants.py.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import visual_constants as _vc


class UIState:
    """Mutable runtime display/inference state; defaults from visual_constants."""

    def __init__(self, args):
        # ── Video ────────────────────────────────────────────────────────
        self.video_index = args.video

        # ── Model Display ────────────────────────────────────────────────
        self.show_superanimal  = True
        self.show_joints       = True
        self.debug_joint_crop  = False

        # ── Custom Joint Model ───────────────────────────────────────────
        self.show_joint_hooves   = _vc.SHOW_JOINT_HOOVES
        self.show_joint_fetlocks = _vc.SHOW_JOINT_FETLOCKS
        self.show_joint_knees    = _vc.SHOW_JOINT_KNEES
        self.show_joint_skeleton = _vc.SHOW_JOINT_SKELETON
        self.joint_color_toggle  = _vc.JOINT_COLOR_TOGGLE

        # ── SuperAnimal ──────────────────────────────────────────────────
        self.show_sa_body     = _vc.SHOW_SA_BODY
        self.show_sa_thais    = _vc.SHOW_SA_THAIS
        self.show_sa_knees    = _vc.SHOW_SA_KNEES
        self.show_sa_paws     = _vc.SHOW_SA_PAWS
        self.show_sa_fetlocks = _vc.SHOW_SA_FETLOCKS
        self.show_sa_skeleton = _vc.SHOW_SA_SKELETON
        self.sa_color_toggle  = _vc.SA_COLOR_TOGGLE

        # ── Keypoint Sizes ───────────────────────────────────────────────
        self.show_bbox        = _vc.SHOW_BBOX
        self.kp_radius        = _vc.KP_RADIUS
        self.kp_radius_joint  = _vc.KP_RADIUS_JOINT
        self.text_scale       = _vc.TEXT_SCALE
        self.line_thickness   = _vc.LINE_THICKNESS

        # ── Inference ────────────────────────────────────────────────────
        self.inference_interval = args.interval
        self.detector_threshold = args.detector_threshold
        self.pose_threshold     = args.pose_threshold
        self.joints_threshold   = args.joints_threshold
        self.frame_pad_factor   = _vc.FRAME_PAD_FACTOR


# ── Widgets ───────────────────────────────────────────────────────────────────

class _CollapsibleSection(tk.Frame):
    def __init__(self, parent, title, **kw):
        super().__init__(parent, bd=1, relief="groove", **kw)
        self._open = True
        self._btn  = tk.Button(
            self, text=f"▼  {title}", anchor="w", relief="flat",
            bg="#dde", font=("TkDefaultFont", 9, "bold"), command=self._toggle,
        )
        self._btn.pack(fill="x")
        self.body = tk.Frame(self, padx=6, pady=4)
        self.body.pack(fill="x")

    def _toggle(self):
        self._open = not self._open
        txt = self._btn.cget("text")
        if self._open:
            self.body.pack(fill="x")
            self._btn.config(text=txt.replace("▶", "▼"))
        else:
            self.body.pack_forget()
            self._btn.config(text=txt.replace("▼", "▶"))


def _check(parent, label, attr, state):
    var = tk.BooleanVar(value=getattr(state, attr))
    def _cb():
        setattr(state, attr, var.get())
    tk.Checkbutton(parent, text=label, variable=var, command=_cb, anchor="w").pack(fill="x")
    return var


def _slider(parent, label, attr, state, lo, hi, res, is_int=False):
    row = tk.Frame(parent)
    row.pack(fill="x", pady=1)
    tk.Label(row, text=label, anchor="w", width=20).pack(side="left")
    disp = tk.StringVar(value=str(getattr(state, attr)))
    tk.Label(row, textvariable=disp, width=5, anchor="e").pack(side="right")
    def _slide(v):
        val = int(float(v)) if is_int else round(float(v), 3)
        setattr(state, attr, val)
        disp.set(str(val))
    s = tk.Scale(row, from_=lo, to=hi, resolution=res,
                 orient="horizontal", command=_slide, showvalue=False)
    s.set(getattr(state, attr))
    s.pack(side="left", fill="x", expand=True)


# ── Control Panel ─────────────────────────────────────────────────────────────

class ControlPanel:
    def __init__(self, root, state, videos):
        root.title("Detection Controls")
        root.resizable(False, True)

        canvas = tk.Canvas(root, width=320)
        sb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))

        # ── Video Selection ──────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "Video")
        sec.pack(fill="x", pady=2, padx=2)
        video_labels = [f"{i}:  {Path(v).name}" for i, v in enumerate(videos)]
        vid_var = tk.StringVar(value=video_labels[state.video_index])
        def _on_video(event=None):
            state.video_index = video_labels.index(vid_var.get())
        combo = ttk.Combobox(sec.body, textvariable=vid_var, values=video_labels,
                             state="readonly", width=30)
        combo.bind("<<ComboboxSelected>>", _on_video)
        combo.pack(fill="x")

        # ── Model Display ────────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "Model Display")
        sec.pack(fill="x", pady=2, padx=2)
        _check(sec.body, "Show SuperAnimal",   "show_superanimal",  state)
        _check(sec.body, "Show Custom Joints", "show_joints",       state)
        _check(sec.body, "Show Bounding Box",  "show_bbox",         state)
        _check(sec.body, "Debug Joint Crop",   "debug_joint_crop",  state)

        # ── SuperAnimal ──────────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "SuperAnimal")
        sec.pack(fill="x", pady=2, padx=2)
        _check(sec.body, "Body (nose / throat / back / tail)", "show_sa_body",     state)
        _check(sec.body, "Thais (upper leg)",                  "show_sa_thais",    state)
        _check(sec.body, "Knees",                              "show_sa_knees",    state)
        _check(sec.body, "Paws",                               "show_sa_paws",     state)
        _check(sec.body, "Fetlocks (from custom model)",       "show_sa_fetlocks", state)
        _check(sec.body, "Skeleton Lines",                     "show_sa_skeleton", state)
        _check(sec.body, "Per-Limb Colors",                    "sa_color_toggle",  state)

        # ── Custom Joints ────────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "Custom Joints")
        sec.pack(fill="x", pady=2, padx=2)
        _check(sec.body, "Hooves",          "show_joint_hooves",   state)
        _check(sec.body, "Fetlocks",        "show_joint_fetlocks", state)
        _check(sec.body, "Knees / Hocks",   "show_joint_knees",    state)
        _check(sec.body, "Skeleton Lines",  "show_joint_skeleton", state)
        _check(sec.body, "Per-Limb Colors", "joint_color_toggle",  state)

        # ── Keypoint Sizes ───────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "Keypoint Sizes")
        sec.pack(fill="x", pady=2, padx=2)
        _slider(sec.body, "SA Radius",      "kp_radius",       state, 1,   20,  1,    is_int=True)
        _slider(sec.body, "Joint Radius",   "kp_radius_joint", state, 1,   20,  1,    is_int=True)
        _slider(sec.body, "Text Scale",     "text_scale",      state, 0.2, 1.5, 0.05)
        _slider(sec.body, "Line Thickness", "line_thickness",  state, 1,   8,   1,    is_int=True)

        # ── Inference ────────────────────────────────────────────────────
        sec = _CollapsibleSection(frame, "Inference")
        sec.pack(fill="x", pady=2, padx=2)
        _slider(sec.body, "Interval (frames)",  "inference_interval", state, 1,  15,  1,    is_int=True)
        _slider(sec.body, "Detector Threshold", "detector_threshold", state, 0,  1,   0.05)
        _slider(sec.body, "Pose Threshold",     "pose_threshold",     state, 0,  1,   0.05)
        _slider(sec.body, "Joints Threshold",   "joints_threshold",   state, 0,  1,   0.01)
        _slider(sec.body, "Frame Pad Factor",   "frame_pad_factor",   state, 0,  1,   0.05)
