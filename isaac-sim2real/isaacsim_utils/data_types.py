"""Shared data types for Isaac Sim motion planning controllers."""

import sys
from dataclasses import dataclass
from pathlib import Path

from curobo.geom.types import Cuboid

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR / "src"))

from common_utils.movesets import SingleRobotMove  # noqa: E402


@dataclass
class PlannedAction:
    """A planned robot action with moves and obstacle cuboids."""

    moves: list[SingleRobotMove]
    obstacles: list[Cuboid]
