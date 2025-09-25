import os
import torch
import logging
from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder
import cv2
import numpy as np

def load_stereo_model(args):
    logging.info("Loading FoundationStereo model...")
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        if k not in cfg: # prevent overriding config from command line
            cfg[k] = args.__dict__[k]
    
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])
    model.cuda().eval()
    return model, args

def run_stereo_inference(model, ir1_np, ir2_np, args):
    ir1_np_bgr = cv2.cvtColor(ir1_np, cv2.COLOR_GRAY2BGR)
    ir2_np_bgr = cv2.cvtColor(ir2_np, cv2.COLOR_GRAY2BGR)
    img0 = cv2.resize(ir1_np_bgr, fx=args.scale, fy=args.scale, dsize=None)
    img1 = cv2.resize(ir2_np_bgr, fx=args.scale, fy=args.scale, dsize=None)
    H_scaled, W_scaled = img0.shape[:2]

    img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
    img0_t, img1_t = padder.pad(img0_t, img1_t)

    with torch.cuda.amp.autocast(True):
        disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)

    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H_scaled, W_scaled)
    
    xx, yy = np.meshgrid(np.arange(W_scaled), np.arange(H_scaled), indexing='xy')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf
    
    return disp, (H_scaled, W_scaled)
