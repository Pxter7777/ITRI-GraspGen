# Installation Guide

## Prerequisite
- Ubuntu-22.04
- Nvidia-driver
- CUDA-12.1
- uv
- git-lfs

#### uv Installation
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
#### git-lfs Installation
```bash
sudo apt install git-lfs
git lfs install
```

## Installation

#### Load CUDA-12.1
- If you're on personal PC, and you do have cuda-12.1 installed at **/usr/local/cuda-12.1**:
    ```bash
    export CUDA_HOME=/usr/local/cuda-12.1
    ```
- else if you're on 140.114... server:
    ```bash
    module unload cuda
    module load cuda/12.1
    ```

#### Clone this repo
```
git clone git@github.com:Pxter7777/ITRI-GraspGen.git
```
#### Update submodules
```bash
git submodule update --init
```

#### Install the venv
```bash
uv sync
```
- This could take a while

#### **Note on `groundingdino/version.py`:**
After running `uv sync`, you might notice an untracked file: `Third_Party/GroundingDINO/groundingdino/version.py`. This file is generated during the installation process of the `groundingdino` submodule. To prevent this file from cluttering your `git status`, you can add it to your local Git exclude list:

```bash
echo "Third_Party/GroundingDINO/groundingdino/version.py" >> .git/info/exclude
```
This command only needs to be run once. It tells Git to ignore the file locally without modifying the submodule's `.gitignore` or the main project's `.gitignore`. This change is local to your repository and will not be committed.

#### Install ZED APK
```bash
<path_to_this_repo>/.venv/bin/python
```


## Download Models
```bash
bash download_model.sh
```