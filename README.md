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

- quick transform
```bash
python manual_PC_transform.py --quick --filename scene_20251014_115650.json --transform-config transform_config_1.json
```