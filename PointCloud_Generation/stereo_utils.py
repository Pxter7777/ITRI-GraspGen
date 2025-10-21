import os
import torch
import logging
from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder
import cv2
import numpy as np


class FoundationStereoModel:
    """A class to encapsulate the FoundationStereo model and its inference."""

    def __init__(self, args: OmegaConf):
        """Initializes the FoundationStereoModel.

        Args:
            args: The configuration arguments.
        """
        logging.info("Loading FoundationStereo model...")
        ckpt_dir = args.ckpt_dir
        cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        for key in args.__dict__:
            if key not in cfg:  # prevent overriding config from command line
                cfg[key] = args.__dict__[key]

        self.args = OmegaConf.create(cfg)
        logging.info(f"args:\n{self.args}")
        logging.info(f"Using pretrained model from {ckpt_dir}")
        self.model = FoundationStereo(self.args)
        ckpt = torch.load(ckpt_dir, weights_only=False)
        logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.model.load_state_dict(ckpt["model"])
        self.model.cuda().eval()

    def run_inference(
        self, ir1_np: np.ndarray, ir2_np: np.ndarray, K: np.ndarray, baseline: float
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Runs stereo inference on a pair of images and returns a depth map.

        Args:
            ir1_np: The left image as a numpy array.
            ir2_np: The right image as a numpy array.
            K: The camera intrinsics matrix.
            baseline: The camera baseline.

        Returns:
            A tuple containing the depth map and the scaled height and width of the images.
        """
        ir1_np_bgr = cv2.cvtColor(ir1_np, cv2.COLOR_GRAY2BGR)
        ir2_np_bgr = cv2.cvtColor(ir2_np, cv2.COLOR_GRAY2BGR)
        img0 = cv2.resize(
            ir1_np_bgr, fx=self.args.scale, fy=self.args.scale, dsize=None
        )
        img1 = cv2.resize(
            ir2_np_bgr, fx=self.args.scale, fy=self.args.scale, dsize=None
        )
        H_scaled, W_scaled = img0.shape[:2]

        img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
        img0_t, img1_t = padder.pad(img0_t, img1_t)

        with torch.cuda.amp.autocast(True):
            disp = self.model.forward(
                img0_t, img1_t, iters=self.args.valid_iters, test_mode=True
            )

        disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H_scaled, W_scaled)

        xx, yy = np.meshgrid(np.arange(W_scaled), np.arange(H_scaled), indexing="xy")
        us_right = xx - disp
        invalid = us_right < 0
        disp[invalid] = np.inf

        # Convert disparity to depth
        fx = K[0, 0] * self.args.scale
        depth = (baseline * fx) / (disp + 1e-6)

        return depth, (H_scaled, W_scaled)
