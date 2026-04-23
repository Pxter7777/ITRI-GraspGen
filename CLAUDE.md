# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

1. **Do NOT run `.py` scripts.** Scripts here are heavy (Isaac Sim, ROS2, large ML models) and require specific live environments. The user runs and terminates them manually.
2. **Do NOT run non-read-only git operations** (`add`, `commit`, `push`, `checkout`, `reset`, `merge`, `rebase`, etc.). Version control is handled manually by the user. `main` only accepts squash-merges from GitHub PRs.
3. **Do NOT edit `Third_Party/`.** These are upstream submodules (GraspGen, SAM2, FoundationStereo, GroundingDINO).
4. **Do NOT attempt mass refactors.** The codebase has known legacy/unused scripts. Stay focused on the requested task.
5. Notes live at `./notes/` (symlink to the user's Obsidian vault).

## Commands

All Python commands go through `uv run` (Python 3.11, `pyproject.toml` pins deps). The user runs scripts; you run lint and tests.

- Lint / format (run before finalizing changes):
  ```bash
  uv run ruff check --fix
  uv run ruff format
  ```
- Tests (pytest, `testpaths = ["tests"]`):
  ```bash
  uv run pytest
  uv run pytest tests/test_your_file.py::test_your_function_name
  uv run pytest -s -v   # show prints
  ```

## Architecture

The grasping pipeline is intentionally split across **three separate processes** that each need a different Python environment, communicating over sockets. Do not try to merge them.

1. **ROS2 server** (`ROS2_server/`) — drives the physical TM arm via `tm_driver`, or a dummy stand-in (`dummy_gripper_server.py`). Runs on system Python with ROS2.
2. **Isaac Sim + cuRobo** (`isaac-sim2real/`) — simulation, collision checking, motion planning. Requires `omni_python` (NVIDIA Omniverse). Entry point: `sync_with_ROS2.py`.
3. **GraspGen workflow** (`scripts/`, `pointcloud_generation/`, `common_utils/`) — 2D→3D pointcloud, SAM2 + GroundingDINO segmentation, grasp pose estimation. Runs under `uv`. Top-level entry: `scripts/workflow_with_isaacsim.py`.

**Inter-process communication** goes through a custom socket module — `common_utils/socket_communication.py` and `isaac-sim2real/isaacsim_utils/socket_communication.py`. Changing a payload or message type has downstream effects across all three processes; audit all sides before editing.

### Package layout

Installed packages (per `pyproject.toml` `[tool.setuptools.packages.find]`): `pointcloud_generation`, `common_utils`, `src`, `ROS2_server`, `Third_Party`. Note that `src/` and `pointcloud_generation/` contain overlapping utilities — treat `pointcloud_generation/` as the current path and `src/` as legacy unless the task says otherwise.

### Logging

For top-level scripts, use the project's formatter for consistent color/format:

```python
import logging
from common_utils.custom_logger import CustomFormatter

handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
```

## Ruff Configuration Notes

`ruff` excludes `Third_Party` and `isaac-sim2real/isaacsim_utils/helper.py`. Ignored rules include `E501` (line length), `I001` (import sorting), `C901` (complexity), `E721`, `E731`, `C408`, `W605` — do not "fix" existing violations of these outside your task scope.

## Temporary Output

Put debug files, dumps, and generated artifacts in `output/`, not the repo root.
