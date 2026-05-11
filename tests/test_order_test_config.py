"""Test the OrderTaskConfig pydantic model for order-based task validation."""

import json

from common_utils.order_task_config import OrderTaskConfig

_ASSETS = (
    "/home/j300/visualize-Fetchbench-assets"
    "/assets/benchmark_objects"
)

_CUP_ID = "dc1c220c8ef89a5d1a57566a9a7e5976"
_TABLE_ID = "d9c11381adf0cf48f1783a44a88d6274"
_STOOL_ID = "1b0626e5a8bf92b3945a77b945b7b70f"
_TEAPOT_ID = "ce04f39420c7c3e82fb82d326efadfe3"
_SUITCASE_ID = "766fe076d4cdef8cf0117851f0671fde"

STANDARD_VALID_JSON = json.dumps(
    {
        "robot_jointstate": [
            1.37296326,
            0.08553859,
            1.05554023,
            2.76803983,
            -1.48792809,
            3.09947786,
        ],
        "target": {
            "instance_id": "Cup_0",
            "category": "Cup",
            "obj_id": _CUP_ID,
            "obj_dir": f"{_ASSETS}/Cup/{_CUP_ID}",
            "obj_type": "mesh",
            "x": 0.64,
            "y": -0.31,
            "z": -0.09,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "scale": 1.5,
            "pose_meter_quat": [
                0.64,
                -0.31,
                -0.09,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
        "obstacles": [
            {
                "instance_id": "Table_0",
                "category": "Table",
                "obj_id": _TABLE_ID,
                "obj_dir": f"{_ASSETS}/Table/{_TABLE_ID}",
                "obj_type": "mesh",
                "x": 0.4,
                "y": -0.7,
                "z": -0.78,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "scale": 10.0,
                "pose_meter_quat": [
                    0.4,
                    -0.7,
                    -0.78,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
            {
                "instance_id": "Stool_0",
                "category": "Stool",
                "obj_id": _STOOL_ID,
                "obj_dir": f"{_ASSETS}/Stool/{_STOOL_ID}",
                "obj_type": "mesh",
                "x": -0.13,
                "y": -0.12,
                "z": -0.55,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "scale": 3.0,
                "pose_meter_quat": [
                    -0.13,
                    -0.12,
                    -0.55,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            },
            {
                "instance_id": "Teapot_0",
                "category": "Teapot",
                "obj_id": _TEAPOT_ID,
                "obj_dir": f"{_ASSETS}/Teapot/{_TEAPOT_ID}",
                "obj_type": "mesh",
                "x": 0.57,
                "y": -0.57,
                "z": -0.15,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 45.0,
                "scale": 1.2,
                "pose_meter_quat": [
                    0.57,
                    -0.57,
                    -0.15,
                    0.0,
                    0.0,
                    0.3826834323650898,
                    0.9238795325112867,
                ],
            },
            {
                "instance_id": "Suitcase_0",
                "category": "Suitcase",
                "obj_id": _SUITCASE_ID,
                "obj_dir": (
                    f"{_ASSETS}/Suitcase/{_SUITCASE_ID}"
                ),
                "obj_type": "mesh",
                "x": 0.48,
                "y": -0.4,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 38.0,
                "scale": 1.0,
                "pose_meter_quat": [
                    0.48,
                    -0.4,
                    0.0,
                    0.0,
                    0.0,
                    0.3255681544571567,
                    0.9455185755993168,
                ],
            },
        ],
    }
)


def test_standard_valid():
    """Validate a fully-populated order task config."""
    data_dict = json.loads(STANDARD_VALID_JSON)
    config = OrderTaskConfig(**data_dict)
    assert config
