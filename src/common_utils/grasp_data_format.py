"""Pydantic models for grasp pose data serialization."""

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

CuroboSuccessType = Literal["Success", "Fail", "Unknown"]


class GraspData(BaseModel):
    """A single grasp pose with its associated metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grasp_pose: np.ndarray  # 4x4

    @field_validator("grasp_pose", mode="before")
    @classmethod
    def coerce_to_ndarray(cls, v: object) -> np.ndarray:
        """Convert input to numpy array before validation."""
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
    """Collection of grasp data entries for a single object."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_grasps: int
    grasps: list[GraspData]
