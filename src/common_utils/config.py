import torch
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_file_dir)
# ------------------- Path Config -------------------
# mostly can be override with args
MODELS_DIR = os.path.join(project_root_dir, "models")
YOLO_CHECKPOINT = os.path.join(MODELS_DIR, "YOLOModels", "yolov5x.pt")
FOUNDATIONSTEREO_CHECKPOINT = os.path.join(
    MODELS_DIR, "FoundationStereoModels", "23-51-11", "model_best_bp2.pth"
)
SAM2_CHECKPOINT = os.path.join(MODELS_DIR, "SAM2Models", "sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GRIPPER_CFG = os.path.join(
    MODELS_DIR, "GraspGenModels", "checkpoints", "graspgen_robotiq_2f_140.yml"
)
GRASPGEN_SCENE_DIR = os.path.join(".", "output")

GROUNDINGDINO_CFG = os.path.join(
    project_root_dir,
    "Third_Party",
    "GroundingDINO",
    "groundingdino",
    "config",
    "GroundingDINO_SwinT_OGC.py",
)
GROUNDINGDINO_CKPT = os.path.join(
    MODELS_DIR, "GroundingDINOModels", "groundingdino_swint_ogc.pth"
)

# ------------------- Other Config -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAY_COLOR_CV = (0, 200, 0)  # light yellow in BGR
OVERLAY_ALPHA = 0.6  # Transparency of the mask
BOX_COLOR = (0, 255, 0)  # Green for the drawing box
