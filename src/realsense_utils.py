import pyrealsense2 as rs
import numpy as np


def intrinsics_from_profile(stream_profile: rs.video_stream_profile):
    intr = stream_profile.get_intrinsics()
    K = np.array(
        [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32
    )
    return intr, K


def extrinsics_Rt(extrin: rs.extrinsics):
    R = np.array(extrin.rotation, dtype=np.float32).reshape(3, 3)
    t = np.array(extrin.translation, dtype=np.float32).reshape(
        3,
    )
    return R, t


def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    W, H, FPS = 848, 480, 6
    config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)

    align_to = rs.stream.infrared
    align = rs.align(align_to)

    ir1_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 1))
    ir2_profile = rs.video_stream_profile(profile.get_stream(rs.stream.infrared, 2))
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))

    _, K_ir1 = intrinsics_from_profile(ir1_profile)
    _, K_color = intrinsics_from_profile(color_profile)
    ext_ir1_to_ir2 = ir1_profile.get_extrinsics_to(ir2_profile)
    ext_ir1_to_color = ir1_profile.get_extrinsics_to(color_profile)

    _, t_ir1_to_ir2 = extrinsics_Rt(ext_ir1_to_ir2)
    baseline = np.linalg.norm(t_ir1_to_ir2)
    K_ir1 = K_ir1.astype(np.float32)

    return pipeline, align, K_ir1, K_color, ext_ir1_to_color, baseline, (W, H)
