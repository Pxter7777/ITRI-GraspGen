# SAM2+FoundationStereo
- TBD
- GenPointCloud
```bash
python main.py --erosion_iterations 0
```
- Zed
```bash
python main_zed.py --erosion_iterations 0
```

- GraspGen
```bash
python ./GraspGen_demo_scene_pc.py --sample_data_dir ./output --gripper_config ./GraspGen/GraspGenModels/checkpoints/graspgen_franka_panda.yml --return_topk --topk_num_grasps 5
```

```bash
python ./scripts/demo_object_mesh.py  --mesh_file /home/j300/GenPointCloud/output/segmented_object_20251003_140757.obj --mesh_scale 1.0 --gripper_config ../GraspGenModels/checkpoints/graspgen_franka_panda.yml 
```
