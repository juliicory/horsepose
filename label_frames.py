import napari
import deeplabcut

CONFIG = r"C:\Users\julic\Documents\GitHub\horsepose\horse_joints-julic-2026-04-29\config.yaml"

deeplabcut.label_frames(CONFIG)
napari.run()
