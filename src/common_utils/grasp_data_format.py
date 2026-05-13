"""Pydantic models for grasp pose data serialization."""

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

CuroboSuccessType = Literal["Success", "Fail", "Unknown"]


class GraspData(BaseModel):
    """A single grasp pose with its associated metadata.

    Attributes:
        model_config: Pydantic model configuration.
        grasp_pose (np.ndarray): 4x4 grasp pose matrix.
        grasp_pose_pre_quat (list[float]): Pre-grasp pose as [x,y,z,qw,qx,qy,qz].
        grasp_pose_quat (list[float]): Grasp pose as [x,y,z,qw,qx,qy,qz].
        curobo_success (CuroboSuccessType): Motion planning result status.
        collision_detected_by_graspgen (bool): Whether GraspGen detected collision.
        distance (float): Distance metric for the grasp.
        horizontal_angle_diff (float): Horizontal angle offset.
        up_vector (float): Up vector component of the grasp.
        discriminator_score (float): Discriminator confidence score.
        motion_plan_time (float): Time taken for motion planning.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grasp_pose: np.ndarray  # 4x4

    @field_validator("grasp_pose", mode="before")
    @classmethod
    def coerce_to_ndarray(cls, v: object) -> np.ndarray:
        """Convert input to numpy array before validation.

        Args:
            v (object): The input value to convert.

        Returns:
            np.ndarray: The converted numpy array.
        """
        return np.array(v)

    grasp_pose_pre_quat: list[float] = Field(..., min_length=7, max_length=7)
    grasp_pose_quat: list[float] = Field(..., min_length=7, max_length=7)
    curobo_success: CuroboSuccessType
    collision_detected_by_graspgen: bool
    distance: float
    horizontal_angle_diff: float
    up_vector: float
    discriminator_score: float
    motion_plan_time: float


class GraspPack(BaseModel):
    """Collection of grasp data entries for a single object.

    Attributes:
        model_config: Pydantic model configuration.
        num_grasps (int): Number of grasps in the collection.
        grasps (list[GraspData]): List of grasp data entries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_grasps: int
    grasps: list[GraspData]
