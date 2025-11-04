import trimesh
import numpy as np


def get_left_up_and_front(grasp: np.array):
    left = grasp[:3, 0]
    up = grasp[:3, 1]
    front = grasp[:3, 2]
    return left, up, front


vec = [
    [0.08023417, 0.0368005, 0.9960965, 0.37843817],
    [0.99140334, 0.1006707, -0.08357546, -0.11308895],
    [-0.10335338, 0.994239, -0.02840698, 0.09745723],
    [0.0, 0.0, 0.0, 1.0],
]

vec = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

grasp = np.array(vec)

quaternion = list(trimesh.transformations.quaternion_from_matrix(grasp))
euler = list(trimesh.transformations.euler_from_matrix(grasp))
print(f"Quaternion: {quaternion}, Euler: {euler}")
left, up, front = get_left_up_and_front(grasp)
print(left, up, front)
euler_orientation = np.rad2deg(euler).tolist()
print(euler_orientation)
# [2025-11-04 10:58:51,073][__main__][INFO] Quaternion: [np.float64(0.5331419630103083), np.float64(0.5020372559690234), np.float64(0.5044905295264966), np.float64(0.4573921146179535)], Euler: [np.float64(1.5839416500506267), np.float64(0.0787557302222831), np.float64(1.4979427454685599)]
