# isaac-sim2real


## IsaacSim Installation Guide

- This part to guide through you installing Isaac Sim 4.5.0 standalone version.
- Official site: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/index.html

### Download
- Download Page: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html
- Isaac Sim 4.5.0 January 2025 Linux (6.7 GB)

### Extract files
```bash
mkdir ~/isaac-sim-4.5.0
unzip isaac-sim-standalone-4.5.0-linux-x86_64.zip -d ~/isaac-sim-4.5.0
```

### About Python Environment
- It's recommended to use the built-in python env IsaacSim provided(which is ./python.sh), so make sure that any other virtual environment(e.g., conda, pyenv, venv) is deactivated.

### Execute post_install.sh
- This only need to be executed once after we extracted the zip file.
```bash
cd ~/isaac-sim-4.5.0 && ./post_install.sh
```

### To Launch IsaacSim
```bash
cd ~/isaac-sim-4.5.0 && ./isaac-sim.sh
```
- First time launching it can take some time(about 3 minutes), it's normal.
---

## cuRobo Installation Guide
- Official Installation Guide: https://curobo.org/get_started/1_install_instructions.html, but since the version isn't matching what I'm using, I recommend to follow my guide below

### Pre-requisite
- CUDA11.8, make sure it's installed.

### Export
- Use CUDA11.8
```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
```

### Add omni_python alias
```bash
echo "alias omni_python='~/isaac-sim-4.5.0/python.sh'" >> ~/.bashrc
```

### Install additional packages
```bash
omni_python -m pip install tomli wheel ninja
```

### Clone
```bash
cd ~ && git clone https://github.com/NVlabs/curobo.git
```

### Install cuRobo
```bash
cd ~/curobo && omni_python -m pip install -e .[isaacsim] --no-build-isolation
```

### Run example
```bash
cd ~/curobo && omni_python examples/isaac_sim/motion_gen_reacher.py
```
- Move the red cube to see the Franka arm folling it.

---
## Copy TM5S configs
### tm_description
```bash
cp -r tm_description ~/curobo/src/curobo/content/assets/robot
```
- It's a bit complicated where tm_description is from, so just copy from my external disk.

### To develop under isaac-sim vscode env
```bash
ln -sf <path_to_isaac-sim2real> <path_to_isaac-sim-4.5.0>/isaac-sim2real
```
e.g.,
```bash
ln -sf /home/j300/ITRI-GraspGen/isaac-sim2real /home/j300/isaac-sim-4.5.0/isaac-sim2real
```

### To make tm5s.yml a symlink for curobo inference:
```bash
ln -sf <path_to_tm5s.yml> <path_to_curobo>/src/curobo/content/configs/robot/tm5s.yml
```
e.g.,
```bash
ln -sf /home/j300/ITRI-GraspGen/isaac-sim2real/tm5s.yml /home/j300/curobo/src/curobo/content/configs/robot/tm5s.yml
```

### To make tm5s.urdf a symlink for curobo inference:
```bash
ln -sf <path_to_tm5s.urdf> <path_to_curobo>/src/curobo/content/assets/robot/tm_description/tm5s.urdf
```
e.g.,
```bash
ln -sf /home/j300/ITRI-GraspGen/isaac-sim2real/tm5s.urdf /home/j300/curobo/src/curobo/content/assets/robot/tm_description/tm5s.urdf
```