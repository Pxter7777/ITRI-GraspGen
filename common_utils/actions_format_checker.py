def is_actions_format_valid(actions) -> bool:
    try:
        if not isinstance(actions, list):
            return False
        for action in actions:
            if not isinstance(action["target_name"], str):
                return False
            if not isinstance(action["qualifier"], str):
                return False
            if not isinstance(action["action"], str):
                return False
            if not isinstance(action["args"], list):
                return False
        return True
    except Exception:
        return False
