import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FETCHBENCH_DIR = "/home/j300/FetchBench-CORL2024-uv/assets/benchmark_objects"


def create_scenario(scenario_id, target_name, target_path, obstacles, output_path):
    logger.info(f"Creating Scenario {scenario_id}...")

    scene_data = {
        "target": {
            "name": target_name,
            "mesh_path": target_path,
            "pose": [0.5, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        },
        "obstacles": {},
        "robot": {
            "config_path": "/home/j300/curobo/src/curobo/content/configs/robot/tm5s.yml",
            "meshes": [
                {
                    "mesh_path": "/home/j300/curobo/src/curobo/content/assets/robot/tm_description/meshes/tm5s/visual/tm5s_base.obj",
                    "pose": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            ],
        },
    }

    offsets = [[0.8, 0.2, 0.0], [0.8, -0.2, 0.0]]
    for i, (name, path) in enumerate(obstacles.items()):
        scene_data["obstacles"][name] = {
            "mesh_path": path,
            "pose": offsets[i],
            "scale": [1.0, 1.0, 1.0],
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_data, f, indent=4)

    logger.info(f"Saved {output_path}")


def main():
    cup_uuid_1 = "106f8b5d2f2c777535c291801eaf5463"
    cup_uuid_2 = "11705ed6a5b1ed37f83d00225df18677"

    bowl_uuid = os.listdir(os.path.join(FETCHBENCH_DIR, "Bowl"))[0]
    bottle_uuid = os.listdir(os.path.join(FETCHBENCH_DIR, "Bottle"))[0]
    bottle_uuid_2 = os.listdir(os.path.join(FETCHBENCH_DIR, "Bottle"))[1]

    cup1_path = os.path.join(FETCHBENCH_DIR, f"Cup/{cup_uuid_1}/mesh.obj")
    cup2_path = os.path.join(FETCHBENCH_DIR, f"Cup/{cup_uuid_2}/mesh.obj")
    bowl_path = os.path.join(FETCHBENCH_DIR, f"Bowl/{bowl_uuid}/mesh.obj")
    bottle_path = os.path.join(FETCHBENCH_DIR, f"Bottle/{bottle_uuid}/mesh.obj")
    bottle_path_2 = os.path.join(FETCHBENCH_DIR, f"Bottle/{bottle_uuid_2}/mesh.obj")

    obstacles_1 = {"obs_bowl": bowl_path, "obs_bottle": bottle_path}

    obstacles_2 = {"obs_bowl": bowl_path, "obs_bottle_2": bottle_path_2}

    create_scenario(
        1,
        "target_cup",
        cup1_path,
        obstacles_1,
        "data_for_test/experiment_scenario_1.json",
    )
    create_scenario(
        2,
        "target_cup",
        cup2_path,
        obstacles_2,
        "data_for_test/experiment_scenario_2.json",
    )


if __name__ == "__main__":
    main()
