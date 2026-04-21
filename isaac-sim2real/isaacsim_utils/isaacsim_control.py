import logging
import sys
import time
import queue
import numpy as np
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from isaacsim_utils.helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR / "src"))

from common_utils.movesets import SingleRobotMove  # noqa: E402
from common_utils.socket_communication import (  # noqa: E402
    NonBlockingJSONSender,
    NonBlockingJSONReceiver,
)
from common_utils import network_config  # noqa: E402

logger = logging.getLogger(__name__)


def cmd_to_move(cmd_plan):
    return cmd_plan.position.cpu().numpy().tolist()


def init_pose_matric(args, motion_gen):
    pose_metric = None
    if args.constrain_grasp_approach:
        pose_metric = PoseCostMetric.create_grasp_approach_metric()
    if args.reach_partial_pose is not None:
        reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
        pose_metric = PoseCostMetric(
            reach_partial_pose=True, reach_vec_weight=reach_vec
        )
    if args.hold_partial_pose is not None:
        hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
        pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
    return pose_metric


def get_cuboid_list(move: SingleRobotMove, obstacles: dict) -> list:
    cuboids = []
    cuboids.append(
        Cuboid(name="table", pose=[0, 0, -1.97] + [1, 0, 0, 0], dims=[4, 4, 4])
    )
    for i, obstacle_name in enumerate(obstacles):
        if obstacle_name not in move.ignore_obstacles:
            middle_point = np.mean(
                [obstacles[obstacle_name]["max"], obstacles[obstacle_name]["min"]],
                axis=0,
            )
            scale = np.array(obstacles[obstacle_name]["max"]) - np.array(
                obstacles[obstacle_name]["min"]
            )
            cuboids.append(
                Cuboid(
                    name=f"obs_{i}",
                    pose=middle_point.tolist() + [1, 0, 0, 0],
                    dims=scale.tolist(),
                )
            )
    return cuboids


def basic_world_config():
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5
    return WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)


def basic_motion_gen(tensor_args, robot_cfg, world_cfg):
    trajopt_tsteps = 32
    trajopt_dt = None
    optimize_dt = True
    trim_steps = None
    interpolation_dt = 0.05
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    return MotionGen(motion_gen_config)


def basic_plan_config():
    return MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=4,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
    )


def zero_obstacle_world_config(usd_help, robot_prim_path):
    return usd_help.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/table",
        ],
    ).get_collision_check_world()


def still_joint_states(joint_states: list, tensor_args: TensorDeviceType, sim_js_names):
    return JointState(
        position=tensor_args.to_device(joint_states),
        velocity=tensor_args.to_device([0.0] * len(joint_states)),
        acceleration=tensor_args.to_device([0.0] * len(joint_states)),
        jerk=tensor_args.to_device([0.0] * len(joint_states)),
        joint_names=sim_js_names,
    )


ROS2StateType = Literal["Ready", "Busy", "Error"]


class IsaacSimController:
    def __init__(self, args, simulation_app) -> None:
        self.args = args
        self.simulation_app = simulation_app
        setup_curobo_logger("warn")

        self.my_world = World(stage_units_in_meters=1.0)
        self.stage = self.my_world.stage
        xform = self.stage.DefinePrim("/World", "Xform")
        self.stage.SetDefaultPrim(xform)
        self.stage.DefinePrim("/curobo", "Xform")
        self.stage = self.my_world.stage

        robot_cfg_path = get_robot_configs_path()
        if args.external_robot_configs_path is not None:
            robot_cfg_path = args.external_robot_configs_path
        robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
        if args.external_asset_path is not None:
            robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
        if args.external_robot_configs_path is not None:
            robot_cfg["kinematics"]["external_robot_configs_path"] = (
                args.external_robot_configs_path
            )
        self.j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

        self.default_config = [
            1.37296326,
            0.08553859,
            1.05554023,
            2.76803983,
            -1.48792809,
            3.09947786,
        ]
        self.robot, self.robot_prim_path = add_robot_to_scene(robot_cfg, self.my_world)

        world_cfg = basic_world_config()
        self.tensor_args = TensorDeviceType()
        self.motion_gen = basic_motion_gen(self.tensor_args, robot_cfg, world_cfg)
        self.plan_config = basic_plan_config()

        print("warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

        print("Curobo is Ready")

        add_extensions(self.simulation_app, self.args.headless_mode)

        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.my_world.stage)
        self.usd_help.add_world_to_stage(world_cfg, base_frame="/World")
        self.zero_obstacles = zero_obstacle_world_config(
            self.usd_help, self.robot_prim_path
        )
        self.pose_metric = init_pose_matric(self.args, self.motion_gen)

        # State tracking queues
        self.planned_action_queue = queue.Queue()
        self.ROS2_fail_queue = queue.Queue()
        self.cmd_plan_positions = None
        self.cmd_idx = 0
        self.articulation_controller = self.robot.get_articulation_controller()
        self.tick = 0
        self.spheres = None
        self.wait_ros2 = False

        # Sockets
        self.graspgen_receiver = NonBlockingJSONReceiver(
            port=network_config.GRASPGEN_TO_ISAACSIM_PORT
        )
        self.graspgen_sender = NonBlockingJSONSender(
            port=network_config.ISAACSIM_TO_GRASPGEN_PORT
        )
        self.ros2_receiver = NonBlockingJSONReceiver(
            port=network_config.ROS2_TO_ISAACSIM_PORT
        )
        self.ros2_sender = NonBlockingJSONSender(
            port=network_config.ISAACSIM_TO_ROS2_PORT
        )

        self.sim_js = self.robot.get_joints_state()
        self.sim_js_names = self.robot.dof_names
        self.planned_action_moves: list[SingleRobotMove] = []
        self.idx_list = [0, 1, 2, 3, 4, 5]
        self.temp_cuboid_paths = []
        self.last_joint_states = self.default_config
        self.common_js_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.graspgen_eof = False
        self.ros2state: ROS2StateType = "Ready"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Exiting due to error: {exc_val}")
        self._close()
        return False

    def _close(self):
        pass

    def _handle_ros2_failure_notices(self):
        if not self.ROS2_fail_queue.empty():
            # clear plan
            while not self.planned_action_queue.empty():
                self.planned_action_queue.get()
            notice = self.ROS2_fail_queue.get()
            print("get Notice", notice)
            if notice["message"] == "Abort":
                self.last_joint_states = self.default_config
                self.graspgen_sender.send_data({"message": "Abort"})
                # eat datas
                for _ in range(5):
                    time.sleep(0.1)
                    self.graspgen_receiver.capture_data()
            elif notice["message"] == "ROS2 Complete":
                if self.graspgen_eof:
                    self.graspgen_sender.send_data({"message": "EOF and ROS2 Complete"})
                else:
                    print("ROS2 complete but not EOF yet.")
            else:
                raise ValueError("Unknown message")

    def _process_graspgen_commands(self, graspgen_datas: list):
        for graspgen_data in graspgen_datas:
            before_move_joints = self.last_joint_states
            processed_moves: list[SingleRobotMove] = []
            for move_dict in graspgen_data["moves"]:
                graspgen_move = SingleRobotMove(**move_dict)
                cuboids = get_cuboid_list(graspgen_move, graspgen_data["obstacles"])

                obstacles = WorldConfig(cuboid=cuboids)
                if graspgen_move.no_obstacles:
                    self.motion_gen.update_world(self.zero_obstacles)
                else:
                    self.motion_gen.update_world(obstacles)

                print("curoboing")
                curobo_cu_js = still_joint_states(
                    self.last_joint_states, self.tensor_args, self.sim_js_names
                )
                if graspgen_move.type == "gripper":
                    processed_moves.append(graspgen_move)
                    continue
                elif graspgen_move.type == "single_pose_meter_quaternion":
                    if graspgen_move.single_pose_meter_quaternion_goal is None:
                        raise ValueError("single_pose_meter_quaternion_goal is missing")
                    ik_goal = Pose(
                        position=self.tensor_args.to_device(
                            graspgen_move.single_pose_meter_quaternion_goal[:3]
                        ),
                        quaternion=self.tensor_args.to_device(
                            graspgen_move.single_pose_meter_quaternion_goal[3:]
                        ),
                    )
                    self.plan_config.pose_cost_metric = self.pose_metric
                    result = self.motion_gen.plan_single(
                        curobo_cu_js.unsqueeze(0), ik_goal, self.plan_config
                    )
                elif graspgen_move.type == "single_pose_joint_rad":
                    joints_goal = JointState(
                        position=self.tensor_args.to_device(
                            graspgen_move.single_pose_joint_rad_goal
                        ),
                        velocity=self.tensor_args.to_device(self.sim_js.velocities)
                        * 0.0,
                        acceleration=self.tensor_args.to_device(self.sim_js.velocities)
                        * 0.0,
                        jerk=self.tensor_args.to_device(self.sim_js.velocities) * 0.0,
                        joint_names=self.sim_js_names,
                    )
                    self.plan_config.pose_cost_metric = self.pose_metric
                    result = self.motion_gen.plan_single_js(
                        curobo_cu_js.unsqueeze(0),
                        joints_goal.unsqueeze(0),
                        self.plan_config,
                    )
                elif graspgen_move.type == "sequence_joint_rad":
                    if (
                        graspgen_move.sequence_joint_rad_goals is None
                        or len(graspgen_move.sequence_joint_rad_goals) == 0
                    ):
                        raise ValueError(
                            f"Can't accept {graspgen_move.sequence_joint_rad_goals} as sequence_joint_rad_goals."
                        )
                    if graspgen_move.no_curobo:
                        processed_moves.append(graspgen_move)
                        self.last_joint_states = graspgen_move.sequence_joint_rad_goals[
                            -1
                        ]
                        continue
                    joints_goal = JointState(
                        position=self.tensor_args.to_device(
                            graspgen_move.sequence_joint_rad_goals[0]
                        ),
                        velocity=self.tensor_args.to_device(self.sim_js.velocities)
                        * 0.0,
                        acceleration=self.tensor_args.to_device(self.sim_js.velocities)
                        * 0.0,
                        jerk=self.tensor_args.to_device(self.sim_js.velocities) * 0.0,
                        joint_names=self.sim_js_names,
                    )
                    self.plan_config.pose_cost_metric = self.pose_metric
                    result = self.motion_gen.plan_single_js(
                        curobo_cu_js.unsqueeze(0),
                        joints_goal.unsqueeze(0),
                        self.plan_config,
                    )

                succ = result.success.item()
                if succ:
                    print("YES YES YES?")
                    new_cmd_plan = result.get_interpolated_plan()
                    new_cmd_plan = self.motion_gen.get_full_js(new_cmd_plan)
                    new_cmd_plan = new_cmd_plan.get_ordered_joint_state(
                        self.common_js_names
                    )
                    if graspgen_move.no_curobo:
                        new_cmd_plan = JointState(
                            position=new_cmd_plan.position[[0, -1]],
                            velocity=new_cmd_plan.velocity[[0, -1]],
                            acceleration=new_cmd_plan.acceleration[[0, -1]],
                            jerk=new_cmd_plan.jerk[[0, -1]],
                            joint_names=new_cmd_plan.joint_names,
                        )
                    positions = cmd_to_move(new_cmd_plan)
                    if graspgen_move.type == "sequence_joint_rad":
                        graspgen_move.sequence_joint_rad_goals = (
                            positions + graspgen_move.sequence_joint_rad_goals
                        )
                    else:
                        graspgen_move.sequence_joint_rad_goals = positions
                    graspgen_move.type = "sequence_joint_rad"
                    processed_moves.append(graspgen_move)
                    self.last_joint_states = positions[-1]
                else:
                    print("This plan failed.")
                    self.last_joint_states = before_move_joints
                    break
            else:
                print("-------------Successfully handled new action--------------")
                self.graspgen_sender.send_data({"message": "Success"})
                self.planned_action_queue.put(
                    {"moves": processed_moves, "obstacles": cuboids}
                )
                break
        else:
            self.graspgen_sender.send_data({"message": "Fail"})

    def _communicate_with_ros2(self) -> None:
        if self.ros2state == "Busy":
            ros2_response: dict = self.ros2_receiver.capture_data()
            if ros2_response is None:
                return
            message = ros2_response.get("message")
            if message == "Success":
                self.ros2state = "Ready"
            elif message == "Fail":
                self.ros2state = "Error"
            else:
                raise ValueError(
                    f"Received unrecongnizable response {ros2_response} from ROS2."
                )

        if self.ros2state == "Ready":
            # Pop a plan if needed and possible
            if (
                len(self.planned_action_moves) == 0
                and not self.planned_action_queue.empty()
            ):
                planned_action = self.planned_action_queue.get()
                self.planned_action_moves = planned_action["moves"]
                if self.temp_cuboid_paths:
                    for path in self.temp_cuboid_paths:
                        self.stage.RemovePrim(path)
                    self.temp_cuboid_paths = []
                for i, cube in enumerate(planned_action["obstacles"]):
                    prim_path = f"/World/temp_obstacle_{i}"
                    cuboid.VisualCuboid(
                        prim_path=prim_path,
                        position=np.array(cube.pose[:3]),
                        scale=np.array(cube.dims),
                        color=np.array([0.0, 0.0, 1.0]),
                    )
                    self.temp_cuboid_paths.append(prim_path)
            # Continue to handle Success message
            if len(self.planned_action_moves) > 0:
                move: SingleRobotMove = self.planned_action_moves.pop(0)
                self.ros2_sender.send_data(asdict(move))
                # For isaac sim animation
                self.cmd_idx = 0
                self.cmd_plan_positions = move.sequence_joint_rad_goals
                self.ros2state = "Busy"
        elif self.ros2state == "Error":
            logger.error(
                "ROS2 failed to move the robot arm, telling graspgen to stop, and resetting JointState"
            )
            # Telling graspgen to stop and eat data
            # Kinda critical, immediately tell graspgen to stop, even though this function isn't supposed to talk to graspgen
            self.graspgen_sender.send_data({"message": "Abort"})
            # eat datas
            for _ in range(5):
                time.sleep(0.1)
                self.graspgen_receiver.capture_data()
            # Reset JointState
            self.last_joint_states = self.default_config
            self.robot.set_joint_positions(self.default_config, self.idx_list)
            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(self.idx_list))]),
                joint_indices=self.idx_list,
            )
            self.cmd_plan_positions = None
            self.planned_action_moves = []
            self.ros2state = "Ready"
        else:
            raise ValueError(
                f"Unexpected logical error, self.ros2state = {self.ros2state}, please enter debug mode to checkout this bug."
            )

    def _communicate_with_graspgen(self) -> None:
        graspgen_datas: list = self.graspgen_receiver.capture_data()

        if graspgen_datas is None:
            return
        elif graspgen_datas[0] == "EOF":
            self.graspgen_eof = True
            return
        elif graspgen_datas[0] == "Reset_to_default":
            self.last_joint_states = self.default_config
            return
        else:
            logger.info("Receiving new action.")
            self._process_graspgen_commands(graspgen_datas)

    def _step_physics_and_visualize(self):
        self.my_world.step(render=True)
        step_index = self.my_world.current_time_step_index

        if not self.my_world.is_playing():
            if self.tick % 100 == 0:
                print("**** Click Play to start simulation *****")
            self.tick += 1
            return

        if step_index < 10:
            self.robot._articulation_view.initialize()
            self.idx_list = [self.robot.get_dof_index(x) for x in self.j_names]
            self.robot.set_joint_positions(self.default_config, self.idx_list)
            self.robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(self.idx_list))]),
                joint_indices=self.idx_list,
            )
        if step_index < 20:
            return

        self.sim_js = self.robot.get_joints_state()
        if self.sim_js is None:
            print("sim_js is None")
            return
        self.sim_js_names = self.robot.dof_names
        if np.any(np.isnan(self.sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = still_joint_states(
            self.sim_js.positions, self.tensor_args, self.sim_js_names
        )

        cu_js.velocity *= 0.0
        cu_js.acceleration *= 0.0
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        if self.args.visualize_spheres and step_index % 2 == 0:
            sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            if self.spheres is None:
                self.spheres = []
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    self.spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        self.spheres[si].set_world_pose(position=np.ravel(s.position))
                        self.spheres[si].set_radius(float(s.radius))

        if self.cmd_plan_positions is not None:
            cmd_state = self.cmd_plan_positions[self.cmd_idx]
            art_action = ArticulationAction(
                cmd_state,
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                joint_indices=self.idx_list,
            )
            self.articulation_controller.apply_action(art_action)
            self.cmd_idx += 1
            for _ in range(2):
                self.my_world.step(render=False)
            if self.cmd_idx >= len(self.cmd_plan_positions):
                self.cmd_idx = 0
                self.cmd_plan_positions = None

    def simulation_loop(self):
        while self.simulation_app.is_running():
            self._communicate_with_ros2()
            self._communicate_with_graspgen()
            self._step_physics_and_visualize()
            # check if idle and eof
            if self.ros2state == "Ready" and self.graspgen_eof:
                self.graspgen_sender.send_data({"message": "EOF and ROS2 Complete"})
                self.graspgen_eof = False
