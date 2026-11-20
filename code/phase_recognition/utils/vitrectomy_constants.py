"""
Single source of truth for the vitrectomy dataset's phase labels and
camera list. Every training and evaluation script should import these
instead of hardcoding them, so that porting the pipeline to a new
dataset (different phases and/or camera count) only requires editing
this file.
"""

CLASS_LABELS = [
    "Room_Prep", "Patient_Entry", "Sterile_Draping", "Microsurgery", "Emergence",
]

DEFAULT_CAMERAS = ["camera01", "camera02", "camera03", "camera04", "camera05"]
