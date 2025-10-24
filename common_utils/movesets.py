import trimesh
import logging
import numpy as np
from common_utils.qualification import get_left_up_and_front

HOME_SIGNAL = [326.8, -140.2, 212.6, 90.0, 0, 90.0]

logger = logging.getLogger(__name__)

def pick_and_pour_and_put_back(grasp: np.array) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p*1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)


    moves.append({"type": "move arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append({"type": "move arm", "goal": HOME_SIGNAL,"wait_time": 0.0})

def grab_and_pour_and_place_back(grasp:np.array, args:list) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p*1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific fixed poses
    ready_pour_pose = args[0]
    pour_pose = args[1]
    before_grasp_position = [p-f*60 for p,f in zip(position, front)]
    grasp_position = [p+f*60 for p,f in zip(position, front)]
    after_grasp_position = grasp_position[:2] + [grasp_position[2]+200]

    release_position = grasp_position[:2] + [grasp_position[2]+5]
    after_release_position = before_grasp_position
    #moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": before_grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "gripper", "goal": "grab"})
    moves.append({"type": "move_arm", "goal": after_grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": ready_pour_pose,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": pour_pose,"wait_time": 1.0})
    moves.append({"type": "move_arm", "goal": ready_pour_pose,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": after_grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": release_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "gripper", "goal": "release"})
    moves.append({"type": "move_arm", "goal": after_release_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    return moves

def grab_and_drop(grasp:np.array, args:list) -> list[dict]:
    moves = []
    # fetch basic infos
    position = grasp[:3, 3].tolist()
    position = [p*1000 for p in position]
    euler_orientation = list(trimesh.transformations.euler_from_matrix(grasp))
    euler_orientation = np.rad2deg(euler_orientation).tolist()
    _, _, front = get_left_up_and_front(grasp)
    front = front.tolist()
    # specific drop point
    drop_pose = args[0]

    before_grasp_position = [p-f*60 for p,f in zip(position, front)]
    grasp_position = [p+f*50 for p,f in zip(position, front)]
    after_grasp_position = grasp_position[:2] + [grasp_position[2]+200]
    forward_signal = HOME_SIGNAL
    forward_signal[0] += 200
    moves.append({"type": "move_arm", "goal": forward_signal,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": before_grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "gripper", "goal": "grab"})
    moves.append({"type": "move_arm", "goal": after_grasp_position + euler_orientation,"wait_time": 0.0})
    moves.append({"type": "move_arm", "goal": drop_pose,"wait_time": 0.5})
    moves.append({"type": "gripper", "goal": "release"})
    moves.append({"type": "move_arm", "goal": HOME_SIGNAL,"wait_time": 0.0})
    return moves

    

action_dict = {
    "grab_and_pour_and_place_back": grab_and_pour_and_place_back,
    "grab_and_drop": grab_and_drop
}

def act(action:str, grasp:np.array, args:list) -> list[dict]:
    if action not in action_dict:
        logger.error(f"There is no such action: {action}")
    action_method = action_dict[action]
    return action_method(grasp, args)
