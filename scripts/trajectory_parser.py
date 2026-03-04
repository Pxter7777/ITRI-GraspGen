import csv
from enum import Enum

class Mode(Enum):
    MOVE = 1
    OPEN = 2
    HALF_OPEN = 3
    CLOSE = 4
    CLOSE_TIGHT = 5

status_open = [0, 0, 0]
status_close = [1, 0, 0]
status_half_open = [0, 1, 0]
status_close_tight = [1, 1, 0]

class Movement:
    def __init__(self, mode, joint_value = None):
        self.mode = mode
        self.joints_values = []
        self.joint_value = joint_value

def load_trajectory_from_csv(filename, delimiter=','):
    try:
        movements = []
        index = 0
        gripper_prev = None
        with open(filename, 'r', newline='') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if index == 0:
                    index += 1
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
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    movements = load_trajectory_from_csv('ROS/trajectories/spoon_peanuts.csv')
    for move in movements:
        print(move.mode)
        print(move.joint_value)