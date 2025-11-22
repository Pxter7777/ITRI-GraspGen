# isaac-sim2real

## To develop under isaac-sim vscode env
```bash
ln -sf <path_to_isaac-sim2real> <path_to_isaac-sim-4.5.0>/isaac-sim2real
```
e.g.,
```bash
ln -sf /home/j300/ITRI-GraspGen/isaac-sim2real /home/j300/isaac-sim-4.5.0/isaac-sim2real
```

## To make tm5s.yml a symlink for curobo inference:
```bash
ln -sf <path_to_tm5s.yml> <path_to_curobo>/src/curobo/content/configs/robot/tm5s.yml
```
e.g.,
```bash
ln -sf /home/j300/isaac-sim-4.5.0/isaac-sim2real/tm5s.yml /home/j300/curobo/src/curobo/content/configs/robot/tm5s.yml
```