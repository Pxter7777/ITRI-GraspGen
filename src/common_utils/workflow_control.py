import logging
import json
import time
from pathlib import Path
from PointCloud_Generation.pointcloud_generation import PointCloudGenerator
from PointCloud_Generation.PC_transform import (
    silent_transform_multiple_obj_with_name_dict,
)

from common_utils.graspgen_utils import GraspGeneratorUI
from common_utils.actions_format_checker import TaskConfig, ObstacleBound
from common_utils.movesets import act_with_name
from common_utils.socket_communication import (
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils.common_utils import create_obstacle_info, load_extra_obstacles
from common_utils.csv_traj_handler import csv_act

logger = logging.getLogger(__name__)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]


class BaseWorkflowController:
    def __init__(
        self, args, sender: NonBlockingJSONSender, receiver: NonBlockingJSONReceiver
    ) -> None:
        self.args = args
        self.sender = sender
        self.receiver = receiver
        self.pc_generator = PointCloudGenerator(self.args)
        self.grasp_generator = GraspGeneratorUI(
            args.gripper_config,
            args.grasp_threshold,
            args.num_grasps,
            args.topk_num_grasps,
            args.need_confirm,
        )
        logger.info("======Successfully initialized======")

    def _send_EOF(self):
        raise NotImplementedError("Child class must implement how to handle EOF")

    def _handle_keyboard_interrupt(self):
        raise NotImplementedError(
            "Child class must implement how to handle keyboard interrupt"
        )

    def _grab_command(self) -> tuple[str, bool]:
        raise NotImplementedError(
            "Child class must implement how to grab command, through CLI input or socket"
        )

    def __enter__(self):
        """Allows the use of 'with GraspGenController(args) as controller:'"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically called when the 'with' block ends.
        Even if an exception occurs, this method runs.
        """
        if exc_type:
            logger.error(f"Exiting due to error: {exc_val}")

        self._close()
        # Returning False allows the exception to continue propagating
        # so you still see the traceback after cleanup.
        return False

    def handle_task_command(self):
        task_type, task_name, no_need_curobo = self._capture_task_name()
        self._eat_aborts()
        if task_type == "GraspGen" and task_name == "Grasp_and_Dump":
            self._process_graspgen_command()
        elif task_type == "csv":
            self._process_csv_command(task_name, no_need_curobo=no_need_curobo)
        else:
            raise ValueError(f"Unknown task_type {task_type}")

    def _process_csv_command(self, command: str, no_need_curobo: bool = False):
        extra_obstacles: dict[str, ObstacleBound] = load_extra_obstacles()
        self._run_csv(command, extra_obstacles, no_need_curobo=no_need_curobo)

    def _run_csv(
        self,
        command: str,
        extra_obstacles: dict[str, ObstacleBound],
        no_need_curobo: bool = False,
    ):
        while True:
            full_acts: list[dict] = csv_act(
                command, extra_obstacles, no_need_curobo=no_need_curobo
            )
            success = self._run_isaacsim(full_acts)
            if success:
                break
            else:
                continue
        self._send_EOF()

    def _process_graspgen_command(self):
        graspgen_filepath = (
            PROJECT_ROOT_DIR / "configs" / "actions" / "Grasp_and_Dump.json"
        )
        task: TaskConfig
        with open(graspgen_filepath, "rb") as f:
            task_json = json.load(f)
            task = TaskConfig(**task_json)
        extra_obstacles: dict = load_extra_obstacles()
        scene_data: dict = self._generate_scene_data(task=task)
        # transform
        scene_data = silent_transform_multiple_obj_with_name_dict(scene_data)
        scene_data = create_obstacle_info(scene_data, extra_obstacles)
        self._run_graspgen(task, scene_data)

    def _run_graspgen(self, task: TaskConfig, scene_data):
        current_target = "Unknown Setup"
        try:
            for move in task.moves:
                current_target = move.target_name
                while True:
                    full_acts: list[dict] = []
                    if move.move_type in [  # Don't need GraspGen
                        "move_to_curobo",
                        "joints_rad_move_to_curobo",
                        "open_grip",
                    ]:
                        full_acts = act_with_name(
                            move.move_type,
                            args=move.args,
                            scene_data=scene_data,
                        )
                    else:  # Need GraspGen
                        grasps = self.grasp_generator.generate_grasp(scene_data, move)
                        full_acts = act_with_name(
                            move.move_type,
                            target_name=move.target_name,
                            grasps=grasps,
                            args=move.args,
                            scene_data=scene_data,
                        )
                    # execute isaacsim
                    success = self._run_isaacsim(full_acts)
                    if success:
                        break
                    else:
                        continue
            self._send_EOF()
        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()
        except InterruptedError as e:
            logger.exception(f"Action for {current_target} interrupted, stopping. {e}")
        except Exception as e:
            logger.exception(
                f"Unknown Error while generating grasp for {current_target}, stopping. {e}"
            )
            raise e

    def _run_isaacsim(self, full_acts: list[dict]) -> bool:
        response = self.receiver.capture_data()
        if response is not None and response["message"] == "Abort":
            raise InterruptedError("aborted by isaacsim, stop current action")
        self.sender.send_data(full_acts)
        # wait for isaacsim's good news
        while response is None:
            response = self.receiver.capture_data()
        if response["message"] == "Success":
            logger.warning("Success")
            return True
        elif response["message"] == "Fail":
            logger.warning("failed")
            return False
        elif response["message"] == "Abort":
            raise InterruptedError("aborted by isaacsim, stop current action")
        else:
            raise ValueError(f"Can't recognize the response {response}")

    def _generate_scene_data(self, task, num_try=20) -> dict:
        # try 20 times
        while True:
            for _ in range(num_try):
                try:
                    scene_data: dict = self.pc_generator.generate_pointcloud(
                        task.track,
                        blockages=task.blockages,
                        valid_region=task.valid_region,
                    )
                    return scene_data
                except ValueError as e:
                    logger.exception(f"{e}, try again")
                    time.sleep(0.1)
                    continue
            else:
                logger.error("Failed to detect using groundingDINO")
                input("Try Again:")

    def _eat_aborts(self):
        # eat aborts if there is any
        for _ in range(5):
            self.receiver.capture_data()

    def _capture_task_name(self) -> tuple[str, str, bool]:
        text, no_need_curobo = self._grab_command()
        if text == "end":
            raise KeyboardInterrupt
        if text == "Grasp_and_Dump":
            return "GraspGen", text, False
        # check if it's in csv
        base_dir = Path("~").expanduser() / "RobotSnackServing-csv"
        if not base_dir.exists():
            raise FileNotFoundError(
                "~/RobotSnackServing is missing. Is https://github.com/hongalicia/RobotSnackServing-csv.git cloned into ~/ ?"
            )
        traj_dir = base_dir / "trajectories"
        csv_paths = list(traj_dir.glob("*.csv"))
        filenames = [path.stem for path in csv_paths]
        if text in filenames:
            return "csv", text, no_need_curobo
        else:
            avail_commands = ["Grasp_and_Dump"] + filenames
            print(
                f"There's no such command called {text}. \nAvailable commands are {avail_commands}"
            )
            return self._capture_task_name()

    def _close(self):
        logger.info("Cleaning up resources...")
        try:
            self.pc_generator.close()
        except Exception as e:
            logger.exception(f"Error during pc_generator cleanup: {e}")
