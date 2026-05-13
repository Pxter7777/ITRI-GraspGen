"""Pydantic models for Isaac Sim order-task configuration."""

import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ObjectBound(BaseModel):
    """Define Obstacle class.

    Attributes:
        model_config: Pydantic model configuration.
        pose_meter_quat (list[float]): 7-element pose [x,y,z,qw,qx,qy,qz].
        instance_id (str): Unique object identifier.
        obj_dir (str): Directory containing the object mesh.
        scale (float): Object scale factor.
        x (float): X position.
        y (float): Y position.
        z (float): Z position.
        roll (float): Roll angle.
        pitch (float): Pitch angle.
        yaw (float): Yaw angle.
    """

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
    """TM5S robot task config with fixed pose and recorded joint state.

    Its pose is fixed.
    We only need to record its jointstate.

    Attributes:
        model_config: Pydantic model configuration.
        robot_jointstate (list[float]): 6-element joint state in radians.
        target (ObjectBound): The target object to grasp.
        obstacles (list[ObjectBound]): Obstacle objects in the scene.
    """

    model_config = ConfigDict(extra="ignore")

    robot_jointstate: list[float] = Field(..., min_length=6, max_length=6)
    target: ObjectBound  # need to be exactly one
    obstacles: list[ObjectBound]  # the count is not limited
