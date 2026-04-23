#!/usr/bin/env python3
"""
SO101 Motion Planning Node
==========================
Wraps MoveIt 2 MoveGroup action into simple callable methods.

PUBLIC API:
  move_to_named_target(name)           → named SRDF pose
  move_to_xyz(x, y, z, ...)           → any XYZ position in base_link
  move_to_pose(pose_stamped)           → full PoseStamped
  open_gripper()                       → open gripper to 1.7453 rad
  close_gripper()                      → close gripper to 0.0 rad
  attach_object()                      → publish True  to /isaac_attach_cube
  detach_object()                      → publish False to /isaac_attach_cube

DESIGN NOTES:
  Threading:
    Runs its own SingleThreadedExecutor in a daemon thread so all public
    methods are safe to call from BT background threads without hitting
    the 'generator already executing' error.

  IK disambiguation:
    A shoulder_pan path constraint is added automatically based on the
    sign of the target Y. This prevents KDL from choosing the mirror-image
    IK solution (arm flipping to the wrong side of the workspace).
      target_y >= 0  →  shoulder_pan hint = +0.3  (arm swings left)
      target_y <  0  →  shoulder_pan hint = -0.3  (arm swings right)

  Start state:
    Every planning request reads fresh joint positions from /joint_states
    and sets them as the explicit start state, preventing MoveIt from
    planning from a stale cached configuration.

  End effector:
    Uses gripper_frame_link (the actual tip, ~10cm below gripper_link).
    This removes a systematic Z offset that appeared when gripper_link
    was used as the control point.
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
    RobotState,
)
from shape_msgs.msg import SolidPrimitive


# ─────────────────────────────────────────────────────────────
# Constants — verified against SRDF and URDF
# ─────────────────────────────────────────────────────────────
ARM_GROUP     = "arm"
GRIPPER_GROUP = "gripper"
ARM_JOINTS    = ["shoulder_pan", "shoulder_lift",
                 "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"

# gripper_frame_link is the actual fingertip frame (~10cm below gripper_link).
# Verified with: ros2 run tf2_ros tf2_echo base_link gripper_frame_link
END_EFFECTOR_LINK = "gripper_frame_link"

WORLD_FRAME = "world"
BASE_FRAME  = "base_link"

# Named poses — values match SRDF exactly
NAMED_POSES = {
    "start": {
        "shoulder_pan": 0.0, "shoulder_lift": 0.0,
        "elbow_flex":   0.0, "wrist_flex":    0.0, "wrist_roll": 0.0,
    },
    "Position_1": {
        "shoulder_pan": 0.7955, "shoulder_lift":  0.5496,
        "elbow_flex":  -0.2521, "wrist_flex":     0.5405, "wrist_roll": 0.0,
    },
    "Position_2": {
        "shoulder_pan": 1.2622, "shoulder_lift": -0.5882,
        "elbow_flex":   0.6256, "wrist_flex":    -0.5954, "wrist_roll": 1.4218,
    },
    "Position_3": {
        "shoulder_pan": -0.5409, "shoulder_lift": -1.3017,
        "elbow_flex":    0.775,  "wrist_flex":     0.0,   "wrist_roll": 0.0,
    },
}

GRIPPER_OPEN_VAL  = 1.7453   # from SRDF 'open'  state
GRIPPER_CLOSE_VAL = 0.0      # from SRDF 'close' state


# ─────────────────────────────────────────────────────────────
# Helper: RPY → quaternion
# ─────────────────────────────────────────────────────────────
def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll  / 2.0);  sr = math.sin(roll  / 2.0)
    cp = math.cos(pitch / 2.0);  sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw   / 2.0);  sy = math.sin(yaw   / 2.0)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


class MotionPlanner(Node):

    def __init__(self):
        super().__init__('so101_motion_planner')

        # Joint state cache — updated by subscriber
        self._joint_lock   = threading.Lock()
        self._joint_states = {}   # name → position (float)

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, 10
        )

        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._attach_pub  = self.create_publisher(Bool, '/isaac_attach_cube', 10)

        # Own executor — runs all ROS callbacks in a separate thread
        self._executor    = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._exec_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name='motion_planner_executor',
        )
        self._exec_thread.start()

        self.get_logger().info('MotionPlanner initialised — waiting for /move_action ...')
        self._move_client.wait_for_server()
        self.get_logger().info('/move_action connected ✓')

        # Wait for first joint state so start_state is immediately available
        self.get_logger().info('Waiting for /joint_states ...')
        deadline = time.time() + 10.0
        while time.time() < deadline:
            with self._joint_lock:
                if self._joint_states:
                    break
            time.sleep(0.05)
        with self._joint_lock:
            ok = bool(self._joint_states)
        if ok:
            self.get_logger().info('/joint_states received ✓')
        else:
            self.get_logger().warn(
                '/joint_states not received in 10s — planning may use stale state'
            )

    # ── Joint state callback ─────────────────────────────────
    def _joint_states_cb(self, msg: JointState):
        with self._joint_lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_states[name] = pos

    def _get_joint_states(self) -> dict:
        with self._joint_lock:
            return dict(self._joint_states)

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        roll:  float = math.pi,   # default: gripper pointing straight down
        pitch: float = 0.0,
        yaw:   float = 0.0,
        frame_id: str = BASE_FRAME,
        constrain_orientation: bool = False,
    ) -> bool:
        """
        Move end-effector tip (gripper_frame_link) to (x, y, z).

        roll=π, pitch=0, yaw=0 (default) = gripper pointing straight down.
        constrain_orientation=False (default): MoveIt picks best IK orientation.
        constrain_orientation=True: enforces the roll/pitch/yaw you pass.
        """
        pose = PoseStamped()
        pose.header.frame_id  = frame_id
        pose.header.stamp     = self.get_clock().now().to_msg()
        pose.pose.position.x  = float(x)
        pose.pose.position.y  = float(y)
        pose.pose.position.z  = float(z)
        pose.pose.orientation = _rpy_to_quaternion(roll, pitch, yaw)

        self.get_logger().info(
            f"move_to_xyz → x={x:.3f}  y={y:.3f}  z={z:.3f}  "
            f"orient_constrained={constrain_orientation}  frame={frame_id}"
        )
        return self._send_pose_goal(ARM_GROUP, pose, constrain_orientation)

    def move_to_pose(self, pose_stamped: PoseStamped,
                     constrain_orientation: bool = False) -> bool:
        """Move end-effector to a full PoseStamped (e.g. from /red_cup_pose)."""
        self.get_logger().info(
            f"move_to_pose → "
            f"x={pose_stamped.pose.position.x:.3f}  "
            f"y={pose_stamped.pose.position.y:.3f}  "
            f"z={pose_stamped.pose.position.z:.3f}  "
            f"frame={pose_stamped.header.frame_id}"
        )
        return self._send_pose_goal(ARM_GROUP, pose_stamped, constrain_orientation)

    def move_to_named_target(self, name: str) -> bool:
        """Move arm to a named joint-space pose defined in the SRDF."""
        if name not in NAMED_POSES:
            self.get_logger().error(
                f"Unknown pose '{name}'. Valid: {list(NAMED_POSES.keys())}"
            )
            return False
        self.get_logger().info(f"move_to_named_target → '{name}'")
        return self._send_joint_goal(ARM_GROUP, ARM_JOINTS, NAMED_POSES[name])

    def open_gripper(self) -> bool:
        """Open gripper to SRDF 'open' value (1.7453 rad)."""
        self.get_logger().info("open_gripper")
        return self._send_joint_goal(
            GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_OPEN_VAL}
        )

    def close_gripper(self) -> bool:
        """Close gripper to SRDF 'close' value (0.0 rad)."""
        self.get_logger().info("close_gripper")
        return self._send_joint_goal(
            GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_CLOSE_VAL}
        )

    def attach_object(self):
        """Tell Isaac Sim to create FixedJoint between jaw and cup."""
        msg = Bool()
        msg.data = True
        self._attach_pub.publish(msg)
        self.get_logger().info("attach_object → /isaac_attach_cube = True")

    def detach_object(self):
        """Tell Isaac Sim to remove FixedJoint between jaw and cup."""
        msg = Bool()
        msg.data = False
        self._attach_pub.publish(msg)
        self.get_logger().info("detach_object → /isaac_attach_cube = False")

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════

    def _build_start_state(self) -> RobotState:
        """
        Build RobotState from latest /joint_states.
        Setting this explicitly prevents MoveIt from using a stale cached
        start state, which causes incorrect IK configurations.
        """
        current = self._get_joint_states()
        rs = RobotState()
        rs.joint_state.header.frame_id = BASE_FRAME
        for jname, jpos in current.items():
            rs.joint_state.name.append(jname)
            rs.joint_state.position.append(jpos)
        return rs

    def _shoulder_pan_hint(self, target_y: float) -> Constraints:
        """
        Loose path constraint on shoulder_pan to guide KDL IK solver
        toward the correct solution family and away from mirror solutions.
          target_y >= 0  →  pan hint = +0.3  (arm swings to +y side)
          target_y <  0  →  pan hint = -0.3  (arm swings to -y side)
        Tolerance ±1.8 rad is loose enough to never block valid solutions.
        """
        jc = JointConstraint()
        jc.joint_name      = "shoulder_pan"
        jc.position        = 0.3 if target_y >= 0.0 else -0.3
        jc.tolerance_above = 1.8
        jc.tolerance_below = 1.8
        jc.weight          = 0.5
        c = Constraints()
        c.joint_constraints.append(jc)
        return c

    def _send_joint_goal(self, group: str, joint_names: list,
                         values: dict) -> bool:
        """Build and send a joint-space goal to MoveIt."""
        constraints = Constraints()
        for jname in joint_names:
            jc = JointConstraint()
            jc.joint_name      = jname
            jc.position        = float(values[jname])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        return self._send_move_goal(group, constraints, path_constraints=None)

    def _send_pose_goal(self, group: str, pose_stamped: PoseStamped,
                        constrain_orientation: bool = False) -> bool:
        """Build and send a Cartesian-space goal to MoveIt."""

        # Position constraint — 2cm sphere around target point
        pos = PositionConstraint()
        pos.header    = pose_stamped.header
        pos.link_name = END_EFFECTOR_LINK
        pos.weight    = 1.0
        region = SolidPrimitive()
        region.type       = SolidPrimitive.SPHERE
        region.dimensions = [0.02]
        pos.constraint_region.primitives.append(region)
        pos.constraint_region.primitive_poses.append(pose_stamped.pose)

        constraints = Constraints()
        constraints.position_constraints.append(pos)

        # Optional orientation constraint
        # Tolerances: tight pitch to keep gripper level, loose roll/yaw
        if constrain_orientation:
            ori = OrientationConstraint()
            ori.header                    = pose_stamped.header
            ori.link_name                 = END_EFFECTOR_LINK
            ori.orientation               = pose_stamped.pose.orientation
            ori.absolute_x_axis_tolerance = 0.3   # roll — some freedom
            ori.absolute_y_axis_tolerance = 0.1   # pitch — keep gripper level
            ori.absolute_z_axis_tolerance = 0.5   # yaw  — wrist_roll handles this
            ori.weight                    = 1.0
            constraints.orientation_constraints.append(ori)

        # Shoulder pan hint to disambiguate IK
        path_constraints = self._shoulder_pan_hint(pose_stamped.pose.position.y)

        return self._send_move_goal(group, constraints,
                                    path_constraints=path_constraints)

    def _send_move_goal(self, group: str, constraints: Constraints,
                        path_constraints=None) -> bool:
        """
        Package constraints into MoveGroup goal, send, and block until done.
        Uses threading.Event so this is safe to call from any thread.
        """
        request = MotionPlanRequest()
        request.group_name                      = group
        request.goal_constraints                = [constraints]
        request.num_planning_attempts           = 10
        request.allowed_planning_time           = 15.0
        request.max_velocity_scaling_factor     = 0.3
        request.max_acceleration_scaling_factor = 0.3

        # Fresh start state from /joint_states
        request.start_state = self._build_start_state()

        # Workspace bounds — slightly larger than SO101 reach
        request.workspace_parameters.header.frame_id = WORLD_FRAME
        request.workspace_parameters.min_corner.x = -0.8
        request.workspace_parameters.min_corner.y = -0.8
        request.workspace_parameters.min_corner.z = -0.2
        request.workspace_parameters.max_corner.x =  0.8
        request.workspace_parameters.max_corner.y =  0.8
        request.workspace_parameters.max_corner.z =  0.8

        if path_constraints is not None:
            request.path_constraints = path_constraints

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay    = 1.0

        done_event       = threading.Event()
        result_container = [None]

        def goal_response_cb(future):
            handle = future.result()
            if not handle.accepted:
                self.get_logger().error("Goal REJECTED by MoveIt")
                result_container[0] = False
                done_event.set()
                return
            self.get_logger().info("Goal accepted — executing ...")
            result_future = handle.get_result_async()
            result_future.add_done_callback(result_cb)

        def result_cb(future):
            code = future.result().result.error_code.val
            if code == 1:
                self.get_logger().info("Motion SUCCEEDED ✓")
                result_container[0] = True
            else:
                self.get_logger().error(
                    f"Motion FAILED — MoveIt error code: {code}  "
                    f"(1=OK -1=FAIL -5=NO_IK -6=TIMEOUT -10=BAD_GROUP)"
                )
                result_container[0] = False
            done_event.set()

        send_future = self._move_client.send_goal_async(goal)
        send_future.add_done_callback(goal_response_cb)

        finished = done_event.wait(timeout=45.0)
        if not finished:
            self.get_logger().error("Motion TIMED OUT after 45s")
            return False
        return bool(result_container[0])


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run: ros2 launch so101_motion_planning motion_planning_test.launch.py
#  OR: ros2 run so101_motion_planning motion_planning_node
# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    planner = MotionPlanner()
    try:
        planner.get_logger().info("=== MOTION PLANNER TEST ===")

        planner.get_logger().info("1. Move to start")
        planner.move_to_named_target('start')
        time.sleep(1.0)

        planner.get_logger().info("2. Open gripper")
        planner.open_gripper()
        time.sleep(1.0)

        planner.get_logger().info("3. Move to Position_1")
        planner.move_to_named_target('Position_1')
        time.sleep(1.0)

        planner.get_logger().info("4. Close gripper")
        planner.close_gripper()
        time.sleep(1.0)

        planner.get_logger().info("5. Move to Position_2")
        planner.move_to_named_target('Position_2')
        time.sleep(1.0)

        planner.get_logger().info("6. Return home")
        planner.move_to_named_target('start')

        planner.get_logger().info("=== TEST COMPLETE ===")
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
