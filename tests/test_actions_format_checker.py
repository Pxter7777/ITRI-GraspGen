import os
from common_utils.actions_format_checker import is_actions_format_valid_v1028

actions_valid = {
    "track": ["green cup", "blue cup", "pan"],
    "actions":
    [
        {
            "target_name": "green cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "grab_and_pour_and_place_back",
            "args": [
                "pan"
            ]
        },
        {
            "target_name": "blue cup",
            "qualifier": "cup_qualifier",
            "action": "move_to",
            "args": [
                [326.8, -140.2, 212.6]
            ]
        }
    ]
    
    
}

def test_is_actions_format_valid_v1028():
    assert is_actions_format_valid_v1028(actions_valid)
    assert not is_actions_format_valid_v1028(None)

def onepone():
    assert True