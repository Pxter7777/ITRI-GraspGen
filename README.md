# SAM2+FoundationStereo
- TBD
- GenPointCloud
```bash
python main.py --erosion_iterations 10
```
- GraspGen
```bash
python ./GraspGen_demo_scene_pc.py --return_topk --topk_num_grasps 10 --filename scene_20251014_115650_transformed.json
```

```bash
python ./scripts/demo_object_mesh.py  --mesh_file /home/j300/GenPointCloud/output/segmented_object_20251003_140757.obj --mesh_scale 1.0 --gripper_config ../GraspGenModels/checkpoints/graspgen_franka_panda.yml 
```

### Proper directory setup
- Home/<username>
    - ITRI-GraspGen(main repo and source code)
    - models
        - FoundationStereoModels
        - GraspGenModels
        - SAM2Models
    - Third_Party
        - FoundationStereo
        - GraspGen
        - sam2

- Idea for dockerization:
    - use bindmount for ITRI-GraspGen(source code) and models
    - bake Third_Party into the image

- quick main_zed
```bash
python main_zed.py --exit-after-save --output-tag "hey"
```

- quick transform
```bash
python manual_PC_transform.py --quick --filename scene_BABA.json --transform-config right.json
```

- quick GraspGen --filename
```bash
python ./GraspGen_demo_scene_pc.py --return_topk --topk_num_grasps 10 --filename scene_change_transformed.json
```

- slow GraspGen
```bash
python ./GraspGen_demo_scene_pc.py --return_topk --topk_num_grasps 10
```

#### Reminder to run tm
- launch this to connect to tm robotiq, before actually run any tm control code.
```bash
ros2 run tm_driver tm_driver robot_ip:=192.168.1.10
```

#### pointcloud_generation.py
- Give it two mode, gui or text mode

#### GroundingDINO test
- Based on the test, using box to cover the metal table can have better result.
- And about all in one or on by one, all in one is actually better.

## Known Issues:
- Manual_Transform_PC will fail if meshcat-server ports get messy.