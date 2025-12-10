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
git submodule update --init --recursive
```

#### Install the venv
```bash
uv sync
```
- This could take a while

#### **Note on `groundingdino/version.py`:**
After running `uv sync`, you might notice an untracked file: `Third_Party/GroundingDINO/groundingdino/version.py`. This file is generated during the installation process of the `groundingdino` submodule. To prevent this file from cluttering your `git status`, you can add it to your local Git exclude list:

```bash
echo "groundingdino/version.py" >> .git/modules/Third_Party/GroundingDINO/info/exclude
```
This command only needs to be run once. It tells Git to ignore the file locally without modifying the submodule's `.gitignore` or the main project's `.gitignore`. This change is local to your repository and will not be committed.

## Install ZED SDK
#### Download
- Go visit [ZED SDK official site](https://www.stereolabs.com/en-fr/developers/release)
- download **ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_\<version\>.zstd.run**
    - Or just copy from my external disk.
#### Install (need sudo)
```bash
./ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_<version>.zstd.run
```

```
To continue you have to accept the EULA. Accept  [Y/n] ? y

ZED SDK will be installed in: /usr/local/zed
[sudo] password for j300: <your_password>

Installing TensorRT 10.9, mandatory dependencies to use the ZED SDK
Install samples (recommended) [Y/n] ? y
Installation path: /usr/local/zed/samples/


Do you want to install the Python API (recommended) [Y/n] ? y
Please specify your python executable: <absolute_path_to_this_repo>/.venv/bin/python
```
- Notice:
    - `ERROR: pip's dependency ...` this can be ignored
    - `To install it later or on a different environment run : 
 python -m pip install --ignore-installed /tmp/<dir_name>/pyzed-5.0-cp311-cp311-linux_x86_64.whl`, go copy the **.whl** file into **./Third_Party**, explain [below](#Reinstall-pyzed-for-your-venv-later)

```
continue...

Do you want to download and optimize the NEURAL Depth models now? These will be required at runtime and will be processed then if not done now, which will extend startup time on first use. [Y/n] ? y

Do you want to run the ZED Diagnostic to download all AI models [Y/n] ? n
```
#### Reinstall pyzed for your venv later
- pyzed can somehow be removed by uv
    - for example, run `uv sync` would delete pyzed because this package isn't managed by uv, and there's no other simple method to prevent this yet. Similar issue see [here](https://github.com/astral-sh/uv/issues/12481))
- If that's the case, run this to reinstall it back.
    ```bash
    uv pip install ./Third_Party/pyzed-5.0-cp311-cp311-linux_x86_64.whl
    ```


## Download Models
```bash
bash download_models.sh
```

## IsaacSim related installation see [here](./isaac-sim2real/README.md)