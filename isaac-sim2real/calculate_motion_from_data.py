import argparse
from omni.isaac.kit import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize_spheres",
        action="store_true",
        help="When True, visualizes robot spheres",
        default=False,
    )
    parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    parser.add_argument(
        "--robot", type=str, default="tm5s.yml", help="robot configuration to load"
    )
    parser.add_argument(
        "--external_asset_path",
        type=str,
        default=None,
        help="Path to external assets when loading an externally located robot",
    )
    parser.add_argument(
        "--external_robot_configs_path",
        type=str,
        default=None,
        help="Path to external robot config when loading an external robot",
    )
    parser.add_argument(
        "--constrain_grasp_approach",
        action="store_true",
        help="When True, approaches grasp with fixed orientation and motion only along z axis.",
        default=False,
    )
    parser.add_argument(
        "--reach_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        help="Reach partial pose",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--hold_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        help="Hold partial pose while moving to goal",
        type=float,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_app = SimulationApp(
        {
            "headless": args.headless_mode is not None,
            "width": "1920",
            "height": "1080",
        }
    )

    from isaacsim_utils.motion_plan_control import MotionPlanController

    with MotionPlanController(args, simulation_app) as controller:
        controller.simulation_loop()


if __name__ == "__main__":
    main()
