import json
import pytest

from pydantic import ValidationError
from common_utils.actions_format_checker import TaskConfig


STANDARD_VALID_JSON = """
{
    "blockages": [
        [915, 242, 1279, 719],
        [413, 205, 500, 300]
    ],
    "valid_region": [0,0,1279,719],
    "track": ["green cup", "blue cup", "glass cup"],
    "extra_obstacles": {
        "pot": {
            "max": [0.06763940904979564, 1.0722684322915268, 0.19641730539823072],
            "min": [-0.25279730730826394, 0.7142005813554249, -0.049177543107872056]
        },
        "waffle_machine": {
            "max": [0.7116523693821937, -0.08480266706786455, 0.23381430176585927],
            "min": [0.4089210481698903, -0.2701176105306099, 0.04335537965711278]
        },
        "splatteu": {
            "max": [0.30579440851407563, 1.032104676590586, 0.17495628388902777],
            "min": [0.22777417701560246, 0.70037035913891, 0.01896955463899678]
        },
        "funnel": {
            "max": [0.9534448779882478, 0.5067593718845549, 0.2276409522244929],
            "min": [0.8184926451424287, 0.39387648388621266, 0.03063070631200454]
        }
    },
    "moves": [
        {
            "target_name": "glass cup",
            "qualifier": "cup_qualifier",
            "move_type": "open_grip",
            "args": [[1.37296326, 0.08553859, 1.05554023, 2.76803983, -1.48792809, 3.09947786]]
        },
        {
            "target_name": "green cup",
            "qualifier": "cup_qualifier",
            "move_type": "grab_and_pour_and_place_back_curobo",
            "args": [[-0.07139691, 0.99580627, 0.15614688]]
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "move_type": "grab_and_pour_and_place_back_curobo",
            "args": [[-0.07139691, 0.99580627, 0.15614688]]
        },
        {
            "target_name": "glass cup",
            "qualifier": "cup_qualifier",
            "move_type": "joints_rad_move_to_curobo",
            "args": [[1.37296326, 0.08553859, 1.05554023, 2.76803983, -1.48792809, 3.09947786]]
        }
    ]
}
"""


def test_standard_valid():
    data_dict = json.loads(STANDARD_VALID_JSON)
    config = TaskConfig(**data_dict)
    assert config


MINIMAL_VALID_JSON = """
{
    "track": ["green cup"],
    "moves": [
        {
            "target_name": "green cup",
            "move_type": "open_grip"
        }
    ]
}
"""


def test_minimal_valid():
    data_dict = json.loads(MINIMAL_VALID_JSON)
    config = TaskConfig(**data_dict)
    assert config


INVALID_TARGET_JSON = """
{
    "track": ["green cup"],
    "moves": [
        {
            "target_name": "glass cup",
            "move_type": "open_grip"
        }
    ]
}
"""


def test_target_not_in_track():
    data_dict = json.loads(INVALID_TARGET_JSON)
    with pytest.raises(ValidationError, match=r"'glass cup' is not in 'track'"):
        _ = TaskConfig(**data_dict)


UNEXPECTED_KEYS_IN_MOVE_JSON = """
{
    "track": ["green cup"],
    "moves": [
        {
            "target_name": "green cup",
            "move_type": "open_grip",
            "BABA_IS_YOU": "BLABLABLA"
        }
    ]
}
"""


def test_unexpected_keys_in_move():
    data_dict = json.loads(UNEXPECTED_KEYS_IN_MOVE_JSON)
    with pytest.raises(
        ValidationError, match=r"BABA_IS_YOU\s+Extra inputs are not permitted"
    ):  # \s stands for any amount of whitespace or newline
        _ = TaskConfig(**data_dict)


UNEXPECTED_KEYS_IN_TASK_JSON = """
{
    "track": ["green cup"],
    "moves": [
        {
            "target_name": "green cup",
            "move_type": "open_grip"
        }
    ],
    "SHORYUKEN": "BLABLABLA"
}
"""


def test_unexpected_keys_in_task():
    data_dict = json.loads(UNEXPECTED_KEYS_IN_TASK_JSON)
    with pytest.raises(
        ValidationError, match=r"SHORYUKEN\s+Extra inputs are not permitted"
    ):
        _ = TaskConfig(**data_dict)


BAD_BLOCKAGE_ELEMENT_NUM_JSON = """
{
    "blockages": [
        [915, 242, 1279, 719, 15],
        [413, 205, 500, 300]
    ],
    "track": ["green cup"],
    "moves": [
        {
            "target_name": "green cup",
            "move_type": "open_grip"
        }
    ]
}
"""


def test_bad_blockage_element_num():
    data_dict = json.loads(BAD_BLOCKAGE_ELEMENT_NUM_JSON)
    with pytest.raises(
        ValidationError,
        match=r"List should have at most 4 items after validation, not 5",
    ):
        _ = TaskConfig(**data_dict)


def main():
    test_standard_valid()
    test_minimal_valid()
    test_target_not_in_track()
    test_unexpected_keys_in_move()
    test_unexpected_keys_in_task()
    test_bad_blockage_element_num()


if __name__ == "__main__":
    main()
