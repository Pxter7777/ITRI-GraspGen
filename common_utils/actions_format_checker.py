import logging
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class ObstacleBound(BaseModel):
    """Define Obstacle class"""

    # max and min must be a list of float with exact 3 elements (X, Y, Z)
    max: List[float] = Field(..., min_length=3, max_length=3)
    min: List[float] = Field(..., min_length=3, max_length=3)


class MoveItem(BaseModel):
    """Define Move class format"""

    # model_config = ConfigDict(extra="allow") # Allow unexpected keys, but this doesn't actually have any effect for now, we accept unexpected keys anyway.

    # Must
    move_type: str

    # Optional
    target_name: Optional[str] = None
    qualifier: Optional[str] = None
    args: Optional[List[Any]] = None


class TaskConfig(BaseModel):
    """Define the whole JSON task file structure"""

    # Must
    moves: List[MoveItem]

    # Optional
    blockages: Optional[List[List[int]]] = None
    track: Optional[List[str]] = None
    extra_obstacles: Optional[Dict[str, ObstacleBound]] = None

    @model_validator(
        mode="after"
    )  # mode='after' means we validate this after all basic type validations.
    def check_target_in_track(self):
        for move in self.moves:
            if move.target_name is not None and move.target_name not in self.track:
                raise ValueError(f"'{move.target_name}' is not in 'track'")
        return self


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
                if isinstance(arg, list) and not (len(arg) == 3 or len(arg) == 6):
                    logger.error(arg)
                    return False
                if isinstance(arg, str) and arg not in actions["track"]:
                    logger.error(arg)
                    return False
        return True
    except Exception as e:
        logger.exception(e)
        return False
