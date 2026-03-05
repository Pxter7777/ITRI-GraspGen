import logging
import csv
import numpy as np
from enum import Enum

"""
Convert given csv file into a full action,
such that it match the json format that is acceptable at the isaac-sim part.
"""

logger = logging.getLogger(__name__)

class Mode(Enum):
    MOVE = 1
    OPEN = 2
    HALF_OPEN = 3
    CLOSE = 4
    CLOSE_TIGHT = 5

class Movement:
    def __init__(self, mode, joint_value = None):
        self.mode = mode
        self.joints_values = []
        self.joint_value = joint_value
status_open = [0, 0, 0]
status_close = [1, 0, 0]
status_half_open = [0, 1, 0]
status_close_tight = [1, 1, 0]

def load_trajectory_from_csv(filename, delimiter=',') -> list[Movement]:
    movements = []
    gripper_prev = None
    # Read csv file
    lines = []
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            lines.append(row)
    
    for index, row in enumerate(lines):
        if index==0:
            continue
        joint_values = (row[0][1:len(row[0])-1]).split(', ')
        joint_values_float = []

        for joint in joint_values:
            joint_values_float.append(float(joint))

        if joint_values_float[6:9] == status_open: # fully open
            if gripper_prev == None or (gripper_prev != None and gripper_prev != status_open):
                move = Movement(Mode.OPEN)                        
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])
        elif joint_values_float[6:9] == status_close: # fully close
            if gripper_prev == None or (gripper_prev != None and gripper_prev != status_close):
                move = Movement(Mode.CLOSE)                        
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])
        elif joint_values_float[6:9] == status_half_open: # half open
            if gripper_prev == None or (gripper_prev != None and gripper_prev != status_half_open):
                move = Movement(Mode.HALF_OPEN)                        
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])
        elif joint_values_float[6:9] == status_close_tight: # half open
            if gripper_prev == None or (gripper_prev != None and gripper_prev != status_close_tight):
                move = Movement(Mode.CLOSE_TIGHT)                        
            else:
                move = Movement(Mode.MOVE, joint_values_float[0:6])
        else:
            move = Movement(Mode.MOVE, joint_values_float[0:6])

        movements.append(move)
        gripper_prev = joint_values_float[6:9]
    return movements

def run_trajectory(filename, vel=40, acc=20, blend=100) -> dict:
    moves = load_trajectory_from_csv(filename)
    parsed_moves: list[Movement] = []
    print("Number of moves:", len(moves))
    last_is_move = False
    for move in moves:
        if move.mode == Mode.MOVE:
            if last_is_move:
                parsed_moves[-1].joints_values.append(list(np.deg2rad(move.joint_value)))
            else:
                parsed_moves.append(move)
                parsed_moves[-1].joints_values.append(list(np.deg2rad(move.joint_value)))
                last_is_move = True
        else:
            last_is_move = False
            parsed_moves.append(move)
    # append positions
    for index, move in enumerate(parsed_moves):
        response = dict()
        if index > 0:
            response["no_curobo"] = true
        if move.mode == Mode.OPEN:
            response = {"type": "gripper", "grip_type": "open", "wait_time": 1.5}
        elif move.mode == Mode.CLOSE:
            response = {"type": "gripper", "grip_type": "close", "wait_time": 0.9}
        elif move.mode == Mode.HALF_OPEN:
            response = {"type": "gripper", "grip_type": "half_open", "wait_time": 0.5}
        elif move.mode == Mode.CLOSE_TIGHT:
            response = {"type": "gripper", "grip_type": "close_tight", "wait_time": 0.9}
        elif move.mode == Mode.MOVE:
            response = {"type": "arm", "joints_values": move.joints_values, "wait_time": 0.0, "custom_vel": vel, "custom_acc": acc, "custom_blend": blend}
    return response


def main():
    csv_example = load_trajectory_from_csv("/home/j300/RobotSnackServing-csv/trajectories/spoon_peanuts.csv")
    for move in csv_example:
        print(move.mode)
        print(move.joint_value)

if __name__=="__main__":
    main()