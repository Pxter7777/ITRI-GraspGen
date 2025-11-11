import numpy as np
from common_utils.graspgen_utils import flip_grasp, flip_upside_down_grasps, get_left_up_and_front

# A standard grasp, oriented with the identity matrix. Up vector is (0, 1, 0).
NORMAL_GRASP = np.array([
    [ 0.05652571, 0.01877481,  0.9982246,   0.45839536],
    [ 0.998081,    0.02425551, -0.05697379,  0.06142139],
    [-0.02528214, 0.9995295,  -0.01736772,  0.08603768],
    [ 0.,          0.,          0.,          1.        ]
], dtype=float)

# An upside-down grasp, rotated 180 degrees around the z-axis. Up vector is (0, -1, 0).
UPSIDEDOWN_GRASP = np.array([
    [-0.66478884, -0.04752735,  0.745518,    0.52082515],
    [-0.7460292,   0.0939154,  -0.6592574,   0.17545402],
    [-0.03868294, -0.99444515, -0.09789073,  0.12312204],
    [ 0.,          0.,          0.,          1.        ]
], dtype=float)

def test_flip_grasp():
    """Tests that flip_grasp rotates the grasp 180 degrees around the front axis."""
    for original_grasp in [NORMAL_GRASP, UPSIDEDOWN_GRASP]:
        grasp_to_flip = original_grasp.copy()
        left_original, up_original, front_original = get_left_up_and_front(grasp_to_flip)

        flipped_grasp = flip_grasp(grasp_to_flip)
        left_new, up_new, front_new = get_left_up_and_front(flipped_grasp)

        # Flipping should negate the left and up vectors, and keep front and position the same.
        np.testing.assert_allclose(left_new, -left_original)
        np.testing.assert_allclose(up_new, -up_original)
        np.testing.assert_allclose(front_new, front_original)
        np.testing.assert_allclose(flipped_grasp[:3, 3], original_grasp[:3, 3])

def test_flip_upside_down_grasps():
    """Tests that only upside-down grasps are flipped."""
    grasps = np.array([NORMAL_GRASP, UPSIDEDOWN_GRASP])
    flipped_grasps = flip_upside_down_grasps(grasps)

    # The normal grasp should remain unchanged.
    np.testing.assert_allclose(flipped_grasps[0], NORMAL_GRASP)

    # The upside-down grasp should be flipped.
    expected_flipped_grasp = flip_grasp(UPSIDEDOWN_GRASP)
    np.testing.assert_allclose(flipped_grasps[1], expected_flipped_grasp)

    # After flipping, the up vector of the formerly upside-down grasp should point upwards.
    _, up_after_flip, _ = get_left_up_and_front(flipped_grasps[1])
    assert up_after_flip[2] > 0