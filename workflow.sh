#!/bin/bash

python=/home/j300/miniforge3/envs/GraspGen/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

$python ./main_zed.py --exit-after-save --output-tag $TIMESTAMP
$python ./manual_PC_transform.py --quick --filename scene_$TIMESTAMP.json --transform-config right2.json
$python ./GraspGen_demo_scene_pc.py --auto-select --return_topk --topk_num_grasps 10 --filename scene_${TIMESTAMP}_transformed.json
/usr/bin/python3 ./gripper.py --input grasp_scene_${TIMESTAMP}_transformed.json
