# Installation Guide

## Prerequisite
- Ubuntu-22.04
- Nvidia-driver
- CUDA-12.1
- uv

#### uv Installation
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
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

#### Install the venv
```bash
uv sync
```
- This could take a while

#### Install ZED APK
```
<path_to_this_repo>/.venv/bin/python
```


## Download Models