import torch

# ------------------- Config -------------------
SAM2_CHECKPOINT = "/home/j300/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OVERLAY_COLOR_CV = (0, 200, 0) # light yellow in BGR
OVERLAY_ALPHA = 0.6 # Transparency of the mask
BOX_COLOR = (0, 255, 0) # Green for the drawing box
