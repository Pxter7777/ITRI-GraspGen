# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

1. **Do NOT run `.py` scripts.** Scripts here are heavy (Isaac Sim, ROS2, large ML models) and require specific live environments. The user runs and terminates them manually.
2. **Do NOT run non-read-only git operations** (`add`, `commit`, `push`, `checkout`, `reset`, `merge`, `rebase`, etc.). Version control is handled manually by the user. `main` only accepts squash-merges from GitHub PRs.
3. **Do NOT edit `third_party/`.** These are upstream submodules (GraspGen, SAM2, FoundationStereo, GroundingDINO).
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

Installed packages (per `pyproject.toml` `[tool.setuptools.packages.find]`): `pointcloud_generation`, `common_utils`, `src`, `ROS2_server`, `third_party`. Note that `src/` and `pointcloud_generation/` contain overlapping utilities — treat `pointcloud_generation/` as the current path and `src/` as legacy unless the task says otherwise.

### Logging

For top-level scripts, use the project's formatter for consistent color/format:

```python
import logging
from common_utils.log_formatter import CustomLoggingFormatter

handler = logging.StreamHandler()
handler.setFormatter(CustomLoggingFormatter())
logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
```

## Code Quality Tools

Three tools enforce code quality. Run all before committing:

1. **Ruff** — linter and formatter. Enforces pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, flake8-bugbear, flake8-annotations, pydocstyle (Google convention), flake8-comprehensions, and ruff-specific rules.
2. **pydoclint** — validates docstring completeness against function signatures.
3. **pyright** — static type checker in strict mode.

CI runs all three on every push and PR.

```bash
uv run ruff check
uv run ruff check --fix
uv run ruff format
uv run pydoclint .
uv run pyright
```

`ruff`, `pydoclint`, and `pyright` all exclude `third_party` and `isaac-sim2real/isaacsim_utils/helper.py`. Do not "fix" existing violations outside your task scope.

## Style Guide

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Key rules not enforced by tooling:

- Use descriptive names over comments to explain intent
- Keep docstrings imperative ("Fetch rows" not "Fetches rows")

### Docstrings (enforced by pydoclint)

- Include `Args:`, `Returns:`, and `Raises:` sections when applicable
- All args/returns must include type hints in docstring: `arg_name (type): Description.`
- `__init__` should NOT have its own docstring — put `Args:` in the class docstring instead
- Class `Attributes:` section must include type hints: `attr_name (type): Description.`

### Type Annotations (enforced by ruff ANN + pyright strict)

- All functions require type annotations for parameters and return values
- Declare class attributes as class-level type annotations (not only in `__init__`)
- Always explicitly annotate class attributes, even when the type is inferrable from the value
- Prefix private/internal class attributes with `_` (these don't need docstring coverage)
- When a string parameter has a known set of allowed values, use `Literal` types instead of bare `str`:

```python
from typing import Literal

Direction = Literal["up", "down", "left", "right"]

def move(direction: Direction) -> None:
    ...
```

- When a third-party function returns an ambiguous type (e.g. `trimesh.load` → `Geometry | list[Geometry]`), prefer an explicit runtime type check over `# type: ignore`. Use `if not isinstance(...): raise TypeError(...)` (not `assert`, which is stripped by `python -O`). Only fall back to `# type: ignore` when the ambiguity is in the return tuple shape or something that can't be narrowed at runtime.

## Temporary Output

Put debug files, dumps, and generated artifacts in `output/`, not the repo root.
