import json
import os
import trimesh
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FETCHBENCH_DIR = "/home/j300/FetchBench-CORL2024-uv/assets/benchmark_objects"

def get_mesh_bounds(mesh_path):
    mesh = trimesh.load(mesh_path)
    # the bounds are [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    bounds = mesh.bounds
    return {
        "min": bounds[0].tolist(),
        "max": bounds[1].tolist()
    }

def create_scenario(scenario_id, target_path, obstacles, output_path):
    logger.info(f"Creating Scenario {scenario_id}...")
    
    # 1. Load target and sample points
    target_mesh = trimesh.load(target_path)
    
    # Optional scaling or translation could go here, but since the target
    # needs to be grasped, let's keep it near origin or scale appropriately
    # We will assume origin is reachable for now. (x=0.5, y=0.0, z=0.0) is a good spot for TM5S
    
    TARGET_OFFSET = np.array([0.5, 0.0, 0.0])
    target_mesh.apply_translation(TARGET_OFFSET)
    
    # Extract points for GraspGen
    points, _ = trimesh.sample.sample_surface(target_mesh, 2048)
    
    scene_data = {
        "scene_info": {
            "pc_color": [] # Will just keep it empty to satisfy parsing
        },
        "object_infos": {
            "target_cup": {
                "points": points.tolist(),
                "colors": np.ones((len(points), 3)).tolist()  # White points
            }
        },
        "obstacles": {}
    }
    
    # 2. Add obstacles as bounding boxes
    offsets = [np.array([0.5, 0.2, 0.0]), np.array([0.5, -0.2, 0.0])]
    for i, (name, path) in enumerate(obstacles.items()):
        obs_mesh = trimesh.load(path)
        obs_mesh.apply_translation(offsets[i])
        bounds = obs_mesh.bounds
        scene_data["obstacles"][name] = {
            "min": bounds[0].tolist(),
            "max": bounds[1].tolist()
        }
    
    # Save the scenario
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_data, f, indent=4)
        
    logger.info(f"Saved {output_path}")

def main():
    # We will pick 1 Cup and 2 Random obstacles (e.g., Bowl, Box/Cabinet)
    
    # Let's get actual UUIDs from FetchBench
    cup_uuid_1 = "106f8b5d2f2c777535c291801eaf5463" # From your ls command
    cup_uuid_2 = "11705ed6a5b1ed37f83d00225df18677"
    
    bowl_uuid = os.listdir(os.path.join(FETCHBENCH_DIR, "Bowl"))[0]
    box_uuid = os.listdir(os.path.join(FETCHBENCH_DIR, "Box"))[0] if "Box" in os.listdir(FETCHBENCH_DIR) else os.listdir(os.path.join(FETCHBENCH_DIR, "Bottle"))[0]
    
    cup1_path = os.path.join(FETCHBENCH_DIR, f"Cup/{cup_uuid_1}/mesh.obj")
    cup2_path = os.path.join(FETCHBENCH_DIR, f"Cup/{cup_uuid_2}/mesh.obj")
    
    bowl_path = os.path.join(FETCHBENCH_DIR, f"Bowl/{bowl_uuid}/mesh.obj")
    
    # Some classes might not exist directly, let's use what we know exists: Bottle
    bottle_uuid = os.listdir(os.path.join(FETCHBENCH_DIR, "Bottle"))[0]
    bottle_path = os.path.join(FETCHBENCH_DIR, f"Bottle/{bottle_uuid}/mesh.obj")

    obstacles_1 = {
        "obstacle_bowl": bowl_path,
        "obstacle_bottle": bottle_path
    }
    
    # Change it up slightly for scenario 2
    bottle_uuid_2 = os.listdir(os.path.join(FETCHBENCH_DIR, "Bottle"))[1]
    bottle_path_2 = os.path.join(FETCHBENCH_DIR, f"Bottle/{bottle_uuid_2}/mesh.obj")
    obstacles_2 = {
        "obstacle_bowl": bowl_path,
        "obstacle_bottle_2": bottle_path_2
    }
    
    create_scenario(1, cup1_path, obstacles_1, "data_for_test/experiment_scenario_1.json")
    create_scenario(2, cup2_path, obstacles_2, "data_for_test/experiment_scenario_2.json")

if __name__ == "__main__":
    main()
