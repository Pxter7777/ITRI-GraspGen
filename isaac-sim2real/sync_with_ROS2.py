import argparse
import sys
from pathlib import Path

# use realpath instead of abspath so we can debug under isaac-sim-4.5.0 folder
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from omni.isaac.kit import SimulationApp  # noqa: E402


def get_headless_mode():
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    peek_args, _ = peek_parser.parse_known_args()
    return peek_args.headless_mode


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
    headless = get_headless_mode()
    simulation_app = SimulationApp(
        {
            "headless": headless is not None,
            "width": "1920",
            "height": "1080",
        }
    )

    args = parse_args()
    from isaacsim_utils.isaacsim_control import IsaacSimController

    with IsaacSimController(args, simulation_app) as controller:
        controller.simulation_loop()


if __name__ == "__main__":
    main()
