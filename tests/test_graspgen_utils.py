import numpy as np
from common_utils.graspgen_utils import flip_grasp, flip_upside_down_grasps, get_left_up_and_front

# A standard grasp, oriented with the identity matrix. Up vector is (0, 1, 0).
NORMAL_GRASP = np.array([
    [1, 0, 0, 0.1],
    [0, 1, 0, 0.2],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1]
], dtype=float)

# An upside-down grasp, rotated 180 degrees around the z-axis. Up vector is (0, -1, 0).
UPSIDEDOWN_GRASP = np.array([
    [-1, 0, 0, 0.1],
    [0, -1, 0, 0.2],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1]
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
    assert up_after_flip[1] > 0