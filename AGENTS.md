# AI Agent Instructions for ITRI-GraspGen

Welcome to the ITRI-GraspGen repository. This document serves as the primary system prompt and instruction manual for any AI coding assistants (like Cursor, Copilot, or CLI agents) operating within this codebase. It contains critical rules, commands, and architectural context required to safely and effectively contribute to the project.

---

## ⚠️ Critical Rules (DO NOT IGNORE)

1. **NO SCRIPT EXECUTION**: Do NOT run any `.py` scripts directly (e.g., `main.py`, `scripts/workflow_with_isaacsim.py`). These scripts are computationally heavy, spin up complex persistent processes (Isaac Sim, ROS2, large ML models), and require specific environments. The human user will handle testing and running scripts manually.
2. **NO MASS REFACTORING**: The project is acknowledged to be complex and contains legacy/needless scripts. Do NOT attempt large-scale architectural reconstructions, file movements, or stylistic overhauls unless explicitly requested. Focus strictly on the immediate, urgent tasks assigned by the user.
3. **NO MAIN BRANCH PUSHES**: Do NOT push directly to the `main` branch. The `main` branch is protected and only accepts merges/squashes from GitHub Pull Requests. Always create a new branch for your work.
4. **NO THIRD-PARTY EDITS**: Do NOT modify files inside the `Third_Party/` directory. These are Git submodules maintained upstream (e.g., GraspGen, SAM2, FoundationStereo, GroundingDINO). 
5. **READ BEFORE EDITING**: Always use the `read`, `glob`, or `grep` tools to verify the contents of files and dependencies before proposing or making changes. Do not hallucinate imports or project structures.
6. **NO GIT OPERATIONS**: Do NOT run `git add`, `git commit`, `git checkout`, `git reset`, etc. Version control should be manually handled by human users.

---

## 🛠️ Build, Lint, and Test Commands

This project uses `uv` for lightning-fast Python dependency management and execution. All Python commands should be prefixed with `uv run` to ensure they execute in the correct virtual environment.

### Linting & Formatting
The project relies on `ruff` for code linting and formatting. Always run these commands before finalizing a task or proposing a commit:

- **Check & Fix Lint Errors (Auto-fixable):**
  ```bash
  uv run ruff check --fix
  ```
- **Format Code:**
  ```bash
  uv run ruff format
  ```
- **Check Lint Errors (Dry Run):**
  ```bash
  uv run ruff check
  ```

### Testing Framework
Tests are managed using `pytest` and are located entirely within the `tests/` directory.

- **Run all tests:**
  ```bash
  uv run pytest
  ```
- **Run a specific test file:**
  ```bash
  uv run pytest tests/test_your_file.py
  ```
- **Run a single test function (Highly Recommended for isolation):**
  ```bash
  uv run pytest tests/test_your_file.py::test_your_function_name
  ```
- **Run tests with standard output (for debugging prints):**
  ```bash
  uv run pytest -s -v
  ```

---

## 📝 Code Style & Guidelines

### 1. Python Version & Typing
- **Target Version**: The project is built for Python 3.11 (`target-version = "py311"`). Use modern Python 3 features accordingly.
- **Type Hinting**: Use extensive type hints (e.g., `list[str]`, `dict[str, Any]`, `int | None`). In a codebase integrating computer vision, robotics, and deep learning, strict typing helps prevent silent tensor/array shape mismatches.
- **Numpy/Torch Types**: When passing around arrays or tensors, clearly document the expected shapes in comments or docstrings (e.g., `# shape: (B, C, H, W)`).

### 2. Formatting Configurations (Ruff Defaults)
- **Line Length**: `ruff` is configured to ignore `E501` (Line too long). Do not aggressively wrap lines or break up strings just to fit an 80-character limit. Readability supersedes strict line limits.
- **Imports**: `ruff` ignores `I001` (Unsorted imports). You do not need to strictly sort imports alphabetized, though keeping standard library, third-party, and local imports grouped logically is best practice.
- **Ignored Rules**: Be aware that `C901` (Function too complex), `E721` (type comparison), and `E731` (assigning lambda) are ignored. Do not try to "fix" existing code that violates these unless it directly relates to your task.

### 3. Naming Conventions
- **Variables & Functions**: `snake_case` (e.g., `process_pointcloud`, `camera_matrix`)
- **Classes**: `PascalCase` (e.g., `GraspEstimator`, `SocketServer`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Private Methods/Variables**: Prefix with an underscore `_` to indicate internal usage.

### 4. Error Handling & Logging
- **Exceptions**: Prefer raising specific built-in exceptions (e.g., `ValueError`, `RuntimeError`, `ConnectionError`) over generic `Exception` types.
- **Network Failures**: This codebase relies heavily on socket communication between different Python processes (ROS2, Isaac Sim, GraspGen). Ensure network errors, timeouts, and disconnects are caught gracefully and logged, rather than crashing the host process.
- **Logging**: Use the standard `logging` module. Do not introduce new logging frameworks. For top-level scripts, set up the root logger using the custom formatter to get preferred color and format:
  ```python
  import logging
  from common_utils.custom_logger import CustomFormatter
  
  handler = logging.StreamHandler()
  handler.setFormatter(CustomFormatter())
  logging.basicConfig(level=logging.DEBUG, handlers=[handler], force=True)
  ```

### 5. File System & Paths
- **Absolute Paths**: Always use absolute paths or resolve relative paths against the project root (`/home/j300/ITRI-GraspGen`).
- **Temporary Output**: When generating temporary debug files, logs, or outputs, place them in the `output/` directory, not in the project root.

---

## 🏗️ Architecture & Component Overview

The pipeline is intentionally decoupled into distinct processes that communicate asynchronously. Do not attempt to merge them into a single monolithic script.

1. **ROS2 Server (`ROS2_server/`)**
   - Manages the physical or dummy robot arm (`tm_driver`).
   - Runs on a specific ROS2 environment.

2. **Isaac Sim + cuRobo (`isaac-sim2real/`)**
   - Handles simulation environments, collision detection, and motion planning.
   - Requires NVIDIA Omniverse Python (`omni_python`).

3. **GraspGen Core (`pointcloud_generation/`, `scripts/`)**
   - The primary data processing workflow.
   - Converts 2D images to 3D pointclouds, runs segmentation (SAM2, GroundingDINO), and estimates grasp poses.

**Communication**: These disparate systems communicate via a custom socket module found at `isaac-sim2real/isaacsim_utils/socket_communication.py` and `common_utils/`. When modifying any API payloads or socket triggers, you must consider the downstream impact across all three systems.

---

## 🔄 Git & Version Control

- **MANUAL ONLY**: As stated in the critical rules, version control is manually handled by human users.
- Do NOT run `git add`, `git commit`, `git push`, `git checkout`, `git reset`, etc., unless explicitly and specifically requested to override this rule by the user.