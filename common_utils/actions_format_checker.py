import logging
from typing import Any, Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


class ObstacleBound(BaseModel):
    """Define Obstacle class"""

    # max and min must be a list of float with exact 3 elements (X, Y, Z)
    max: list[float] = Field(..., min_length=3, max_length=3)
    min: list[float] = Field(..., min_length=3, max_length=3)


class MoveItem(BaseModel):
    """Define Move class format"""

    model_config = ConfigDict(extra="forbid")  # Forbid unexpected keys.

    # Must
    move_type: str

    # Optional
    target_name: str | None = None
    qualifier: str | None = None
    args: list[Any] | None = None


FourIntList = Annotated[list[int], Field(min_length=4, max_length=4)]


class TaskConfig(BaseModel):
    """Define the whole JSON task file structure"""

    model_config = ConfigDict(extra="forbid")  # Forbid unexpected keys.

    # Must
    moves: list[MoveItem]

    # Optional
    blockages: list[FourIntList] = Field(default_factory=list)
    track: list[str] = Field(default_factory=list)
    extra_obstacles: dict[str, ObstacleBound] = Field(default_factory=dict)
    valid_region: FourIntList | None = None

    @model_validator(
        mode="after"
    )  # mode="after" means we validate this after all basic type validations.
    def check_target_in_track(self):
        for move in self.moves:
            if move.target_name is None:
                continue
            if self.track is None:
                raise ValueError(f"'target_name' is {move.target_name} yet 'track' is None")
            if move.target_name not in self.track:
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
