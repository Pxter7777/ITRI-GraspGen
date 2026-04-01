import logging
import numpy as np
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

CuroboSuccessType = Literal["Success", "Fail", "Unknown"]


class GraspData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grasp_pose: np.ndarray  # 4x4
    grasp_pose_pre_quat: list[float] = Field(..., min_length=7, max_length=7)
    grasp_pose_quat: list[float] = Field(..., min_length=7, max_length=7)
    curobo_success: CuroboSuccessType
    collision_detected_by_graspgen: bool


class GraspPack(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    num_grasps: int
    grasps: list[GraspData]
