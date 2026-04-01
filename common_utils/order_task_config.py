import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ObjectBound(BaseModel):
    """Define Obstacle class"""

    model_config = ConfigDict(extra="ignore")

    pose_meter_quat: list[float] = Field(..., min_length=7, max_length=7)
    instance_id: str
    obj_dir: str
    scale: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class OrderTaskConfig(BaseModel):
    """
    For tm5s robot config
    its pose is fixed.
    we only need to record its jointstate.
    """

    model_config = ConfigDict(extra="ignore")

    robot_jointstate: list[float] = Field(..., min_length=6, max_length=6)
    target: ObjectBound  # need to be exactly one
    obstacles: list[ObjectBound]  # the count is not limited
