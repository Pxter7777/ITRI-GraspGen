import logging

logger = logging.getLogger(__name__)

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
def is_actions_format_valid_v1028(actions) -> bool:
    try:
        if not isinstance(actions["track"], list):
            logger.error(actions["track"])
            return False
        for track in actions["track"]:
            if not isinstance(track, str):
                logger.error(track)
                return False
        if not isinstance(actions["actions"], list):
            logger.error(actions["actions"])
            return False
        for action in actions["actions"]:
            if not isinstance(action["target_name"], str):
                logger.error(action["target_name"])
                return False
            if not isinstance(action["qualifier"], str):
                logger.error(action["qualifier"])
                return False
            if not isinstance(action["action"], str):
                logger.error(action["action"])
                return False
            if not isinstance(action["args"], list):
                logger.error(action["args"])
                return False
            for arg in action["args"]:
                logger.info(arg)
                if not (isinstance(arg, list) or isinstance(arg, str)):
                    logger.error(arg)
                    return False
                if isinstance(arg, list) and len(arg)!=3:
                    logger.error(arg)
                    return False
                if isinstance(arg, str) and arg not in actions["track"]:
                    logger.error(arg)
                    return False
        return True
    except Exception as e:
        logger.exception(e)
        return False
