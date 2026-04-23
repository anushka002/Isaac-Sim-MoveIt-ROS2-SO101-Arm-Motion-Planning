#!/usr/bin/env python3
"""
SO101 Motion Planning Node
==========================
Wraps MoveIt 2 MoveGroup action into simple callable methods.

PUBLIC API:
  move_to_named_target(name)           → named SRDF pose
  move_to_xyz(x, y, z, ...)           → any XYZ position
  move_to_pose(pose_stamped)           → full PoseStamped
  open_gripper()                       → open gripper
  close_gripper()                      → close gripper
  attach_object()                      → Isaac attach
  detach_object()                      → Isaac detach

THREADING FIX:
  MotionPlanner spins its OWN executor in a background thread.
  This means rclpy.spin(bt_node) and MotionPlanner calls never
  conflict — they live in completely separate executors.
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Bool
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PositionConstraint,
    OrientationConstraint,
)
from shape_msgs.msg import SolidPrimitive


# ─────────────────────────────────────────────────────────────
# Constants — from SRDF
# ─────────────────────────────────────────────────────────────
ARM_GROUP         = "arm"
GRIPPER_GROUP     = "gripper"
ARM_JOINTS        = ["shoulder_pan", "shoulder_lift",
                     "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT     = "gripper"
END_EFFECTOR_LINK = "gripper_link"
WORLD_FRAME       = "world"
BASE_FRAME        = "base_link"

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

GRIPPER_OPEN_VAL  = 1.7453
GRIPPER_CLOSE_VAL = 0.0


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
    """
    Drop-in motion planning library for SO101.

    Threading model:
    ────────────────
    This node runs its OWN SingleThreadedExecutor in a daemon thread.
    All ROS callbacks (action responses, topic callbacks) are handled
    there — completely separate from whatever node calls this planner.

    This means you can safely call any method from:
      - A BT background thread
      - The main thread
      - Any other thread
    without hitting the 'generator already executing' error.
    """

    def __init__(self):
        super().__init__('so101_motion_planner')

        self._move_client = ActionClient(self, MoveGroup, '/move_action')
        self._attach_pub  = self.create_publisher(Bool, '/isaac_attach_cube', 10)

        # ── Own executor in own thread ────────────────────────
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._exec_thread = threading.Thread(
            target=self._executor.spin,
            daemon=True,
            name='motion_planner_executor',
        )
        self._exec_thread.start()
        # ─────────────────────────────────────────────────────

        self.get_logger().info('MotionPlanner ready — waiting for /move_action ...')
        self._move_client.wait_for_server()
        self.get_logger().info('/move_action connected ✓')

    # ═══════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════

    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        roll:  float = math.pi,
        pitch: float = 0.0,
        yaw:   float = 0.0,
        frame_id: str = BASE_FRAME,
        constrain_orientation: bool = False,
    ) -> bool:
        """
        Move end-effector to (x, y, z).

        constrain_orientation=False (default):
            MoveIt picks whatever orientation solves IK best.
            Recommended — avoids IK failures on SO101.

        constrain_orientation=True:
            Forces the roll/pitch/yaw you specify.
            Use sparingly.
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
            f"constrain={constrain_orientation}  frame={frame_id}"
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
        """Move arm to a named joint-space pose from the SRDF."""
        if name not in NAMED_POSES:
            self.get_logger().error(
                f"Unknown named pose '{name}'. Valid: {list(NAMED_POSES.keys())}"
            )
            return False
        self.get_logger().info(f"move_to_named_target → '{name}'")
        return self._send_joint_goal(ARM_GROUP, ARM_JOINTS, NAMED_POSES[name])

    def open_gripper(self) -> bool:
        self.get_logger().info("open_gripper")
        return self._send_joint_goal(
            GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_OPEN_VAL}
        )

    def close_gripper(self) -> bool:
        self.get_logger().info("close_gripper")
        return self._send_joint_goal(
            GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_CLOSE_VAL}
        )

    def attach_object(self):
        msg = Bool()
        msg.data = True
        self._attach_pub.publish(msg)
        self.get_logger().info("attach_object → /isaac_attach_cube = True")

    def detach_object(self):
        msg = Bool()
        msg.data = False
        self._attach_pub.publish(msg)
        self.get_logger().info("detach_object → /isaac_attach_cube = False")

    # ═══════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ═══════════════════════════════════════════════════════════

    def _send_joint_goal(self, group, joint_names, values):
        constraints = Constraints()
        for jname in joint_names:
            jc = JointConstraint()
            jc.joint_name      = jname
            jc.position        = float(values[jname])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            constraints.joint_constraints.append(jc)
        return self._send_move_goal(group, constraints)

    def _send_pose_goal(self, group: str, pose_stamped: PoseStamped,
                        constrain_orientation: bool = False) -> bool:
        pos = PositionConstraint()
        pos.header    = pose_stamped.header
        pos.link_name = END_EFFECTOR_LINK
        pos.weight    = 1.0
        region = SolidPrimitive()
        region.type       = SolidPrimitive.SPHERE
        region.dimensions = [0.02]   # 2cm tolerance
        pos.constraint_region.primitives.append(region)
        pos.constraint_region.primitive_poses.append(pose_stamped.pose)

        constraints = Constraints()
        constraints.position_constraints.append(pos)

        if constrain_orientation:
            ori = OrientationConstraint()
            ori.header                    = pose_stamped.header
            ori.link_name                 = END_EFFECTOR_LINK
            ori.orientation               = pose_stamped.pose.orientation
            ori.absolute_x_axis_tolerance = 0.2
            ori.absolute_y_axis_tolerance = 0.2
            ori.absolute_z_axis_tolerance = 0.2
            ori.weight                    = 1.0
            constraints.orientation_constraints.append(ori)

        return self._send_move_goal(group, constraints)

    def _send_move_goal(self, group: str, constraints: Constraints) -> bool:
        """
        Send goal to MoveIt and block until result.

        Uses threading.Event instead of spin_until_future_complete so this
        method can safely be called from ANY thread — the node's own executor
        (running in _exec_thread) handles all the ROS callbacks.
        """
        request = MotionPlanRequest()
        request.group_name                      = group
        request.goal_constraints                = [constraints]
        request.num_planning_attempts           = 5
        request.allowed_planning_time           = 10.0
        request.max_velocity_scaling_factor     = 0.3
        request.max_acceleration_scaling_factor = 0.3

        request.workspace_parameters.header.frame_id = WORLD_FRAME
        request.workspace_parameters.min_corner.x = -1.5
        request.workspace_parameters.min_corner.y = -1.5
        request.workspace_parameters.min_corner.z = -0.5
        request.workspace_parameters.max_corner.x =  1.5
        request.workspace_parameters.max_corner.y =  1.5
        request.workspace_parameters.max_corner.z =  2.5

        goal = MoveGroup.Goal()
        goal.request = request
        goal.planning_options.plan_only       = False
        goal.planning_options.replan          = True
        goal.planning_options.replan_attempts = 3
        goal.planning_options.replan_delay    = 2.0

        # threading.Event lets us block the calling thread
        # while the executor thread handles ROS callbacks
        done_event      = threading.Event()
        result_container = [None]   # [True/False]

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
                self.get_logger().error(f"Motion FAILED — error code: {code}")
                result_container[0] = False
            done_event.set()

        send_future = self._move_client.send_goal_async(goal)
        send_future.add_done_callback(goal_response_cb)

        # Block the calling thread (BT background thread) until done
        # Timeout = 30s — covers planning + execution time
        finished = done_event.wait(timeout=30.0)
        if not finished:
            self.get_logger().error("Motion TIMED OUT after 30s")
            return False

        return bool(result_container[0])


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run: ros2 run so101_motion_planning motion_planning_node
# ─────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    planner = MotionPlanner()

    try:
        planner.get_logger().info("=== MOTION PLANNER TEST ===")

        planner.close_gripper()
        time.sleep(0.5)

        planner.move_to_named_target('start')
        time.sleep(1.0)

        planner.open_gripper()
        time.sleep(1.0)

        planner.move_to_xyz(x=0.15, y=0.25, z=0.20)
        time.sleep(1.0)

        planner.move_to_xyz(x=0.15, y=0.25, z=0.12)
        time.sleep(1.0)

        planner.close_gripper()
        time.sleep(0.5)

        planner.attach_object()
        time.sleep(0.5)

        planner.move_to_xyz(x=0.15, y=0.25, z=0.25)
        time.sleep(1.0)

        planner.move_to_xyz(x=0.20, y=-0.14, z=0.20)
        time.sleep(1.0)

        planner.detach_object()
        time.sleep(0.5)

        planner.open_gripper()
        time.sleep(1.0)

        planner.move_to_named_target('start')
        planner.get_logger().info("=== TEST COMPLETE ===")

    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()









# #!/usr/bin/env python3
# """
# SO101 Motion Planning Node
# ==========================
# Wraps MoveIt 2 MoveGroup action into simple callable methods.

# PUBLIC API — these are the only functions you need to call:
# ───────────────────────────────────────────────────────────
#   move_to_named_target(name)
#       Move arm to a pose defined in the SRDF by name.
#       Names: 'start', 'Position_1', 'Position_2', 'Position_3'

#   move_to_xyz(x, y, z, roll, pitch, yaw, frame_id, constrain_orientation)
#       Move end-effector to any position in space.
#       By default constrain_orientation=False — MoveIt picks best IK solution.
#       Set constrain_orientation=True only if you need a specific grasp angle.

#   move_to_pose(pose_stamped)
#       Move end-effector to a full PoseStamped (e.g. from /red_cup_pose).

#   open_gripper()       → open gripper
#   close_gripper()      → close gripper
#   attach_object()      → tell Isaac to lock cup to gripper jaw
#   detach_object()      → tell Isaac to release cup

# USAGE EXAMPLE (in bt_node.py or anywhere):
# ──────────────────────────────────────────
#   from so101_motion_planning.motion_planner import MotionPlanner

#   planner = MotionPlanner()

#   # Normal move — MoveIt picks best orientation (no IK failures)
#   planner.move_to_xyz(x=0.15, y=0.25, z=0.20)

#   # Force a specific orientation (use sparingly)
#   planner.move_to_xyz(x=0.15, y=0.25, z=0.15,
#                       roll=0.0, pitch=math.pi/2, yaw=0.0,
#                       constrain_orientation=True)

#   planner.move_to_pose(red_cup_pose_stamped)
#   planner.move_to_named_target('start')
# """

# import math
# import time

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from geometry_msgs.msg import PoseStamped, Quaternion
# from std_msgs.msg import Bool
# from moveit_msgs.action import MoveGroup
# from moveit_msgs.msg import (
#     MotionPlanRequest,
#     Constraints,
#     JointConstraint,
#     PositionConstraint,
#     OrientationConstraint,
# )
# from shape_msgs.msg import SolidPrimitive


# # ─────────────────────────────────────────────────────────────
# # Constants — from SRDF / robot description
# # ─────────────────────────────────────────────────────────────
# ARM_GROUP         = "arm"
# GRIPPER_GROUP     = "gripper"
# ARM_JOINTS        = ["shoulder_pan", "shoulder_lift",
#                      "elbow_flex", "wrist_flex", "wrist_roll"]
# GRIPPER_JOINT     = "gripper"
# END_EFFECTOR_LINK = "gripper_link"
# WORLD_FRAME       = "world"
# BASE_FRAME        = "base_link"

# NAMED_POSES = {
#     "start": {
#         "shoulder_pan": 0.0, "shoulder_lift": 0.0,
#         "elbow_flex":   0.0, "wrist_flex":    0.0, "wrist_roll": 0.0,
#     },
#     "Position_1": {
#         "shoulder_pan": 0.7955, "shoulder_lift":  0.5496,
#         "elbow_flex":  -0.2521, "wrist_flex":     0.5405, "wrist_roll": 0.0,
#     },
#     "Position_2": {
#         "shoulder_pan": 1.2622, "shoulder_lift": -0.5882,
#         "elbow_flex":   0.6256, "wrist_flex":    -0.5954, "wrist_roll": 1.4218,
#     },
#     "Position_3": {
#         "shoulder_pan": -0.5409, "shoulder_lift": -1.3017,
#         "elbow_flex":    0.775,  "wrist_flex":     0.0,   "wrist_roll": 0.0,
#     },
# }

# GRIPPER_OPEN_VAL  = 1.7453
# GRIPPER_CLOSE_VAL = 0.0


# # ─────────────────────────────────────────────────────────────
# # Helper: convert roll/pitch/yaw → quaternion
# # ─────────────────────────────────────────────────────────────
# def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
#     cr = math.cos(roll  / 2.0);  sr = math.sin(roll  / 2.0)
#     cp = math.cos(pitch / 2.0);  sp = math.sin(pitch / 2.0)
#     cy = math.cos(yaw   / 2.0);  sy = math.sin(yaw   / 2.0)
#     q = Quaternion()
#     q.w = cr * cp * cy + sr * sp * sy
#     q.x = sr * cp * cy - cr * sp * sy
#     q.y = cr * sp * cy + sr * cp * sy
#     q.z = cr * cp * sy - sr * sp * cy
#     return q


# class MotionPlanner(Node):

#     def __init__(self):
#         super().__init__('so101_motion_planner')
#         self._move_client = ActionClient(self, MoveGroup, '/move_action')
#         self._attach_pub  = self.create_publisher(Bool, '/isaac_attach_cube', 10)
#         self.get_logger().info('MotionPlanner ready — waiting for /move_action ...')
#         self._move_client.wait_for_server()
#         self.get_logger().info('/move_action connected ✓')

#     # ═══════════════════════════════════════════════════════════
#     # PUBLIC API
#     # ═══════════════════════════════════════════════════════════

#     def move_to_xyz(
#         self,
#         x: float,
#         y: float,
#         z: float,
#         roll:  float = math.pi,
#         pitch: float = 0.0,
#         yaw:   float = 0.0,
#         frame_id: str = BASE_FRAME,
#         constrain_orientation: bool = False,  # False = let MoveIt pick best IK
#     ) -> bool:
#         """
#         Move the end-effector to (x, y, z).

#         constrain_orientation=False (default):
#             MoveIt picks whatever orientation solves IK best.
#             For SO101 this will naturally be roughly horizontal — ideal for
#             side-grasping a mug. No IK failures.

#         constrain_orientation=True:
#             Forces the gripper to the roll/pitch/yaw you specify.
#             Use sparingly — can cause IK failures on low-DOF arms.
#         """
#         pose = PoseStamped()
#         pose.header.frame_id  = frame_id
#         pose.header.stamp     = self.get_clock().now().to_msg()
#         pose.pose.position.x  = float(x)
#         pose.pose.position.y  = float(y)
#         pose.pose.position.z  = float(z)
#         pose.pose.orientation = _rpy_to_quaternion(roll, pitch, yaw)

#         self.get_logger().info(
#             f"move_to_xyz → x={x:.3f}  y={y:.3f}  z={z:.3f}  "
#             f"constrain_orientation={constrain_orientation}  frame={frame_id}"
#         )
#         return self._send_pose_goal(ARM_GROUP, pose, constrain_orientation)

#     def move_to_pose(self, pose_stamped: PoseStamped,
#                      constrain_orientation: bool = False) -> bool:
#         """Move end-effector to a full PoseStamped (e.g. from /red_cup_pose)."""
#         self.get_logger().info(
#             f"move_to_pose → "
#             f"x={pose_stamped.pose.position.x:.3f}  "
#             f"y={pose_stamped.pose.position.y:.3f}  "
#             f"z={pose_stamped.pose.position.z:.3f}  "
#             f"frame={pose_stamped.header.frame_id}"
#         )
#         return self._send_pose_goal(ARM_GROUP, pose_stamped, constrain_orientation)

#     def move_to_named_target(self, name: str) -> bool:
#         """Move arm to a named joint-space pose from the SRDF."""
#         if name not in NAMED_POSES:
#             self.get_logger().error(
#                 f"Unknown named pose '{name}'. Valid: {list(NAMED_POSES.keys())}"
#             )
#             return False
#         self.get_logger().info(f"move_to_named_target → '{name}'")
#         return self._send_joint_goal(ARM_GROUP, ARM_JOINTS, NAMED_POSES[name])

#     def open_gripper(self) -> bool:
#         self.get_logger().info("open_gripper")
#         return self._send_joint_goal(
#             GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_OPEN_VAL}
#         )

#     def close_gripper(self) -> bool:
#         self.get_logger().info("close_gripper")
#         return self._send_joint_goal(
#             GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_CLOSE_VAL}
#         )

#     def attach_object(self):
#         msg = Bool(); msg.data = True
#         self._attach_pub.publish(msg)
#         self.get_logger().info("attach_object → /isaac_attach_cube = True")

#     def detach_object(self):
#         msg = Bool(); msg.data = False
#         self._attach_pub.publish(msg)
#         self.get_logger().info("detach_object → /isaac_attach_cube = False")

#     # ═══════════════════════════════════════════════════════════
#     # PRIVATE HELPERS
#     # ═══════════════════════════════════════════════════════════

#     def _send_joint_goal(self, group, joint_names, values):
#         constraints = Constraints()
#         for jname in joint_names:
#             jc = JointConstraint()
#             jc.joint_name      = jname
#             jc.position        = float(values[jname])
#             jc.tolerance_above = 0.01
#             jc.tolerance_below = 0.01
#             jc.weight          = 1.0
#             constraints.joint_constraints.append(jc)
#         return self._send_move_goal(group, constraints)

#     def _send_pose_goal(self, group: str, pose_stamped: PoseStamped,
#                         constrain_orientation: bool = False) -> bool:
#         """
#         Build a Cartesian-space goal.
#         Orientation constraint is OPTIONAL — omitted by default so MoveIt
#         freely picks the best IK solution (natural horizontal grasp on SO101).
#         """
#         # Position constraint — 2 cm sphere (slightly looser = more IK solutions)
#         pos = PositionConstraint()
#         pos.header    = pose_stamped.header
#         pos.link_name = END_EFFECTOR_LINK
#         pos.weight    = 1.0
#         region = SolidPrimitive()
#         region.type       = SolidPrimitive.SPHERE
#         region.dimensions = [0.02]
#         pos.constraint_region.primitives.append(region)
#         pos.constraint_region.primitive_poses.append(pose_stamped.pose)

#         constraints = Constraints()
#         constraints.position_constraints.append(pos)

#         # Only add orientation constraint if explicitly requested
#         if constrain_orientation:
#             ori = OrientationConstraint()
#             ori.header                    = pose_stamped.header
#             ori.link_name                 = END_EFFECTOR_LINK
#             ori.orientation               = pose_stamped.pose.orientation
#             ori.absolute_x_axis_tolerance = 0.2
#             ori.absolute_y_axis_tolerance = 0.2
#             ori.absolute_z_axis_tolerance = 0.2
#             ori.weight                    = 1.0
#             constraints.orientation_constraints.append(ori)

#         return self._send_move_goal(group, constraints)

#     def _send_move_goal(self, group, constraints):
#         request = MotionPlanRequest()
#         request.group_name                      = group
#         request.goal_constraints                = [constraints]
#         request.num_planning_attempts           = 5
#         request.allowed_planning_time           = 10.0
#         request.max_velocity_scaling_factor     = 0.3
#         request.max_acceleration_scaling_factor = 0.3

#         request.workspace_parameters.header.frame_id = WORLD_FRAME
#         request.workspace_parameters.min_corner.x = -1.5
#         request.workspace_parameters.min_corner.y = -1.5
#         request.workspace_parameters.min_corner.z = -0.5
#         request.workspace_parameters.max_corner.x =  1.5
#         request.workspace_parameters.max_corner.y =  1.5
#         request.workspace_parameters.max_corner.z =  2.5

#         goal = MoveGroup.Goal()
#         goal.request = request
#         goal.planning_options.plan_only       = False
#         goal.planning_options.replan          = True
#         goal.planning_options.replan_attempts = 3
#         goal.planning_options.replan_delay    = 2.0

#         future = self._move_client.send_goal_async(goal)
#         rclpy.spin_until_future_complete(self, future)

#         handle = future.result()
#         if not handle.accepted:
#             self.get_logger().error("Goal REJECTED by MoveIt")
#             return False

#         result_future = handle.get_result_async()
#         rclpy.spin_until_future_complete(self, result_future)

#         code = result_future.result().result.error_code.val
#         if code == 1:
#             self.get_logger().info("Motion SUCCEEDED ✓")
#             return True
#         else:
#             self.get_logger().error(f"Motion FAILED — error code: {code}")
#             return False


# # ─────────────────────────────────────────────────────────────
# # STANDALONE TEST
# # Run: ros2 run so101_motion_planning motion_planning_node
# # ─────────────────────────────────────────────────────────────
# def main(args=None):
#     rclpy.init(args=args)
#     planner = MotionPlanner()

#     try:
#         planner.get_logger().info("=== MOTION PLANNER TEST ===")

#         # Step 0: close gripper initially
#         planner.close_gripper()
#         time.sleep(0.5)

#         # Step 1: go home
#         planner.move_to_named_target('start')
#         time.sleep(1.0)

#         # Step 2: open gripper
#         planner.open_gripper()
#         time.sleep(1.0)

#         # Step 3: pre-grasp — 20cm above cup
#         # constrain_orientation=False → MoveIt picks natural horizontal grasp
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.20)
#         time.sleep(1.0)
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.15,
#                       roll=0.0, pitch=math.pi/2, yaw=0.0,
#                       constrain_orientation=True)

#         # Step 4: descend to grasp height
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.12)
#         time.sleep(1.0)

#         # Step 5: close gripper
#         planner.close_gripper()
#         time.sleep(0.5)

#         # Step 6: attach in Isaac Sim
#         planner.attach_object()
#         time.sleep(0.5)

#         # Step 7: lift straight up
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.20)
#         time.sleep(1.0)

#         # Step 8: move to bin
#         planner.move_to_xyz(x=0.20, y=-0.25, z=0.12)
#         time.sleep(1.0)

#         # Step 9: detach in Isaac Sim
#         planner.detach_object()
#         time.sleep(0.5)

#         # Step 10: open gripper
#         planner.open_gripper()
#         time.sleep(1.0)

#         # Step 11: return home
#         planner.move_to_named_target('start')

#         planner.get_logger().info("=== TEST COMPLETE ===")

#     except KeyboardInterrupt:
#         pass
#     finally:
#         planner.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()



























# WORKED - ONLY MOVEMENTS / GRIPPER ORIENTATION NTOT FIXED.



# #!/usr/bin/env python3
# """
# SO101 Motion Planning Node
# ==========================
# Wraps MoveIt 2 MoveGroup action into simple callable methods.

# PUBLIC API — these are the only functions you need to call:
# ───────────────────────────────────────────────────────────
#   move_to_named_target(name)
#       Move arm to a pose defined in the SRDF by name.
#       Names: 'start', 'Position_1', 'Position_2', 'Position_3'

#   move_to_xyz(x, y, z, roll, pitch, yaw, frame_id)
#       Move end-effector to any position in space.
#       You pass raw numbers — no PoseStamped needed.
#       roll/pitch/yaw default to pointing straight down.

#   move_to_pose(pose_stamped)
#       Move end-effector to a full PoseStamped.
#       Used when you already have a PoseStamped (e.g. from /red_cup_pose).

#   open_gripper()       → open gripper
#   close_gripper()      → close gripper
#   attach_object()      → tell Isaac to lock cup to gripper jaw
#   detach_object()      → tell Isaac to release cup

# USAGE EXAMPLE (in bt_node.py or anywhere):
# ──────────────────────────────────────────
#   from so101_motion_planning.motion_planner import MotionPlanner

#   planner = MotionPlanner()

#   # Move to a specific XYZ point (e.g. above the red cup)
#   planner.move_to_xyz(x=0.236, y=0.31, z=0.20)   # pre-grasp: 20cm above cup

#   # Move to the cup itself
#   planner.move_to_xyz(x=0.236, y=0.31, z=0.05)   # grasp height

#   # Move using a PoseStamped from perception
#   planner.move_to_pose(red_cup_pose_stamped)

#   # Move to named pose from SRDF
#   planner.move_to_named_target('start')
# """

# import math
# import time

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from geometry_msgs.msg import PoseStamped, Quaternion
# from std_msgs.msg import Bool
# from moveit_msgs.action import MoveGroup
# from moveit_msgs.msg import (
#     MotionPlanRequest,
#     Constraints,
#     JointConstraint,
#     PositionConstraint,
#     OrientationConstraint,
# )
# from shape_msgs.msg import SolidPrimitive


# # ─────────────────────────────────────────────────────────────
# # Constants — from SRDF / robot description
# # ─────────────────────────────────────────────────────────────
# ARM_GROUP         = "arm"
# GRIPPER_GROUP     = "gripper"
# ARM_JOINTS        = ["shoulder_pan", "shoulder_lift",
#                      "elbow_flex", "wrist_flex", "wrist_roll"]
# GRIPPER_JOINT     = "gripper"
# END_EFFECTOR_LINK = "gripper_link"
# WORLD_FRAME       = "world"
# BASE_FRAME        = "base_link"

# # Named poses — joint values copied directly from SRDF
# NAMED_POSES = {
#     "start": {
#         "shoulder_pan": 0.0, "shoulder_lift": 0.0,
#         "elbow_flex":   0.0, "wrist_flex":    0.0, "wrist_roll": 0.0,
#     },
#     "Position_1": {
#         "shoulder_pan": 0.7955, "shoulder_lift":  0.5496,
#         "elbow_flex":  -0.2521, "wrist_flex":     0.5405, "wrist_roll": 0.0,
#     },
#     "Position_2": {
#         "shoulder_pan": 1.2622, "shoulder_lift": -0.5882,
#         "elbow_flex":   0.6256, "wrist_flex":    -0.5954, "wrist_roll": 1.4218,
#     },
#     "Position_3": {
#         "shoulder_pan": -0.5409, "shoulder_lift": -1.3017,
#         "elbow_flex":    0.775,  "wrist_flex":     0.0,   "wrist_roll": 0.0,
#     },
# }

# GRIPPER_OPEN_VAL  = 1.7453  # radians — from SRDF 'open'  state
# GRIPPER_CLOSE_VAL = 0.0     # radians — from SRDF 'close' state


# # ─────────────────────────────────────────────────────────────
# # Helper: convert roll/pitch/yaw → quaternion
# # ─────────────────────────────────────────────────────────────
# def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
#     """
#     Convert Euler angles (radians) to a ROS Quaternion.
#     Used so you can specify gripper orientation as human-readable angles.

#     Default orientation (roll=π, pitch=0, yaw=0) = gripper pointing straight DOWN.
#     This is the standard top-down grasp orientation.
#     """
#     cr = math.cos(roll  / 2.0)
#     sr = math.sin(roll  / 2.0)
#     cp = math.cos(pitch / 2.0)
#     sp = math.sin(pitch / 2.0)
#     cy = math.cos(yaw   / 2.0)
#     sy = math.sin(yaw   / 2.0)

#     q = Quaternion()
#     q.w = cr * cp * cy + sr * sp * sy
#     q.x = sr * cp * cy - cr * sp * sy
#     q.y = cr * sp * cy + sr * cp * sy
#     q.z = cr * cp * sy - sr * sp * cy
#     return q


# class MotionPlanner(Node):
#     """
#     Drop-in motion planning library for SO101.
#     Import this class and call its methods — no MoveIt boilerplate needed.
#     """

#     def __init__(self):
#         super().__init__('so101_motion_planner')

#         # MoveIt action client — sends goals to /move_action
#         self._move_client = ActionClient(self, MoveGroup, '/move_action')

#         # Isaac Sim attach/detach publisher
#         self._attach_pub = self.create_publisher(Bool, '/isaac_attach_cube', 10)

#         self.get_logger().info('MotionPlanner ready — waiting for /move_action ...')
#         self._move_client.wait_for_server()
#         self.get_logger().info('/move_action connected ✓')

#     # ═══════════════════════════════════════════════════════════
#     # PUBLIC API
#     # ═══════════════════════════════════════════════════════════

#     def move_to_xyz(
#         self,
#         x: float,
#         y: float,
#         z: float,
#         roll:  float = math.pi,   # default: gripper pointing straight down
#         pitch: float = 0.0,
#         yaw:   float = 0.0,
#         frame_id: str = BASE_FRAME,
#     ) -> bool:
#         """
#         Move the end-effector to any point in space.

#         Parameters
#         ----------
#         x, y, z   : target position in metres (in frame_id frame)
#         roll      : gripper roll  in radians  (default π = pointing down)
#         pitch     : gripper pitch in radians  (default 0)
#         yaw       : gripper yaw   in radians  (default 0)
#         frame_id  : reference frame           (default 'base_link')

#         Returns True on success, False on failure.

#         Examples
#         --------
#         # Pre-grasp — 15 cm above cup, gripper pointing down
#         planner.move_to_xyz(0.236, 0.31, 0.20)

#         # Grasp — at cup height
#         planner.move_to_xyz(0.236, 0.31, 0.05)

#         # Place — above box
#         planner.move_to_xyz(0.10, -0.20, 0.15)

#         # Custom orientation — gripper tilted 45 degrees in pitch
#         planner.move_to_xyz(0.3, 0.0, 0.1, roll=math.pi, pitch=math.pi/4)
#         """
#         pose = PoseStamped()
#         pose.header.frame_id    = frame_id
#         pose.header.stamp       = self.get_clock().now().to_msg()
#         pose.pose.position.x    = float(x)
#         pose.pose.position.y    = float(y)
#         pose.pose.position.z    = float(z)
#         pose.pose.orientation   = _rpy_to_quaternion(roll, pitch, yaw)

#         self.get_logger().info(
#             f"move_to_xyz → x={x:.3f}  y={y:.3f}  z={z:.3f}  "
#             f"rpy=({roll:.2f}, {pitch:.2f}, {yaw:.2f})  frame={frame_id}"
#         )
#         return self._send_pose_goal(ARM_GROUP, pose)

#     def move_to_pose(self, pose_stamped: PoseStamped) -> bool:
#         """
#         Move end-effector to a full PoseStamped.

#         Use this when you already have a PoseStamped — e.g. from /red_cup_pose.

#         Example
#         -------
#         # In your BT leaf, after receiving the cup pose:
#         planner.move_to_pose(self.cup_pose)   # cup_pose is a PoseStamped
#         """
#         self.get_logger().info(
#             f"move_to_pose → "
#             f"x={pose_stamped.pose.position.x:.3f}  "
#             f"y={pose_stamped.pose.position.y:.3f}  "
#             f"z={pose_stamped.pose.position.z:.3f}  "
#             f"frame={pose_stamped.header.frame_id}"
#         )
#         return self._send_pose_goal(ARM_GROUP, pose_stamped)

#     def move_to_named_target(self, name: str) -> bool:
#         """
#         Move arm to a named joint-space pose from the SRDF.

#         Valid names: 'start', 'Position_1', 'Position_2', 'Position_3'

#         Use for: home position, known waypoints, safe intermediate poses.
#         """
#         if name not in NAMED_POSES:
#             self.get_logger().error(
#                 f"Unknown named pose '{name}'. "
#                 f"Valid: {list(NAMED_POSES.keys())}"
#             )
#             return False

#         self.get_logger().info(f"move_to_named_target → '{name}'")
#         return self._send_joint_goal(ARM_GROUP, ARM_JOINTS, NAMED_POSES[name])

#     def open_gripper(self) -> bool:
#         """Open gripper (1.7453 rad)."""
#         self.get_logger().info("open_gripper")
#         return self._send_joint_goal(
#             GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_OPEN_VAL}
#         )

#     def close_gripper(self) -> bool:
#         """Close gripper (0.0 rad)."""
#         self.get_logger().info("close_gripper")
#         return self._send_joint_goal(
#             GRIPPER_GROUP, [GRIPPER_JOINT], {GRIPPER_JOINT: GRIPPER_CLOSE_VAL}
#         )

#     def attach_object(self):
#         """Tell Isaac Sim to attach cup to gripper jaw (FixedJoint ON)."""
#         msg = Bool()
#         msg.data = True
#         self._attach_pub.publish(msg)
#         self.get_logger().info("attach_object → /isaac_attach_cube = True")

#     def detach_object(self):
#         """Tell Isaac Sim to release cup from gripper jaw (FixedJoint OFF)."""
#         msg = Bool()
#         msg.data = False
#         self._attach_pub.publish(msg)
#         self.get_logger().info("detach_object → /isaac_attach_cube = False")

#     # ═══════════════════════════════════════════════════════════
#     # PRIVATE HELPERS
#     # ═══════════════════════════════════════════════════════════

#     def _send_joint_goal(
#         self, group: str, joint_names: list, values: dict
#     ) -> bool:
#         """Build a joint-space goal and send to MoveIt."""
#         constraints = Constraints()
#         for jname in joint_names:
#             jc = JointConstraint()
#             jc.joint_name      = jname
#             jc.position        = float(values[jname])
#             jc.tolerance_above = 0.01
#             jc.tolerance_below = 0.01
#             jc.weight          = 1.0
#             constraints.joint_constraints.append(jc)

#         return self._send_move_goal(group, constraints)

#     def _send_pose_goal(self, group: str, pose_stamped: PoseStamped) -> bool:
#         """Build a Cartesian-space goal and send to MoveIt."""

#         # Position constraint — 1 cm sphere around the target point
#         pos = PositionConstraint()
#         pos.header    = pose_stamped.header
#         pos.link_name = END_EFFECTOR_LINK
#         pos.weight    = 1.0
#         region = SolidPrimitive()
#         region.type       = SolidPrimitive.SPHERE
#         region.dimensions = [0.01]          # 1 cm tolerance
#         pos.constraint_region.primitives.append(region)
#         pos.constraint_region.primitive_poses.append(pose_stamped.pose)

#         # Orientation constraint — loose tolerance so IK has room to solve
#         ori = OrientationConstraint()
#         ori.header       = pose_stamped.header
#         ori.link_name    = END_EFFECTOR_LINK
#         ori.orientation  = pose_stamped.pose.orientation
#         ori.absolute_x_axis_tolerance = 0.3   # radians
#         ori.absolute_y_axis_tolerance = 0.3
#         ori.absolute_z_axis_tolerance = 0.3
#         ori.weight = 1.0

#         constraints = Constraints()
#         constraints.position_constraints.append(pos)
#         constraints.orientation_constraints.append(ori)

#         return self._send_move_goal(group, constraints)

#     def _send_move_goal(self, group: str, constraints: Constraints) -> bool:
#         """Package constraints into a MoveGroup goal, send, and wait."""
#         request = MotionPlanRequest()
#         request.group_name                      = group
#         request.goal_constraints                = [constraints]
#         request.num_planning_attempts           = 5
#         request.allowed_planning_time           = 10.0
#         request.max_velocity_scaling_factor     = 0.3
#         request.max_acceleration_scaling_factor = 0.3

#         # Workspace — bounding box MoveIt plans inside
#         request.workspace_parameters.header.frame_id = WORLD_FRAME
#         request.workspace_parameters.min_corner.x = -1.5
#         request.workspace_parameters.min_corner.y = -1.5
#         request.workspace_parameters.min_corner.z = -0.5
#         request.workspace_parameters.max_corner.x =  1.5
#         request.workspace_parameters.max_corner.y =  1.5
#         request.workspace_parameters.max_corner.z =  2.5

#         goal = MoveGroup.Goal()
#         goal.request = request
#         goal.planning_options.plan_only       = False   # plan + execute
#         goal.planning_options.replan          = True
#         goal.planning_options.replan_attempts = 3
#         goal.planning_options.replan_delay    = 2.0

#         # Send and block until done
#         future = self._move_client.send_goal_async(goal)
#         rclpy.spin_until_future_complete(self, future)

#         handle = future.result()
#         if not handle.accepted:
#             self.get_logger().error("Goal REJECTED by MoveIt")
#             return False

#         result_future = handle.get_result_async()
#         rclpy.spin_until_future_complete(self, result_future)

#         code = result_future.result().result.error_code.val
#         if code == 1:
#             self.get_logger().info("Motion SUCCEEDED ✓")
#             return True
#         else:
#             # Common error codes:
#             # -1=FAILURE  -5=NO_IK_SOLUTION  -6=TIMED_OUT  -10=INVALID_GROUP_NAME
#             self.get_logger().error(f"Motion FAILED — error code: {code}")
#             return False


# # ─────────────────────────────────────────────────────────────
# # STANDALONE TEST
# # Run:  ros2 run so101_motion_planning motion_planner
# # ─────────────────────────────────────────────────────────────
# def main(args=None):
#     rclpy.init(args=args)
#     planner = MotionPlanner()

#     try:
#         planner.get_logger().info("=== MOTION PLANNER TEST ===")

#         # --- Step 0 : Close gripper initially
#         planner.close_gripper()
#         time.sleep(0.5)

#         # ── Step 1: go home
#         planner.move_to_named_target('start')
#         time.sleep(1.0)

#         # ── Step 2: open gripper
#         planner.open_gripper()
#         time.sleep(1.0)

#         # ── Step 3: pre-grasp — 18cm above cup
#         # Cup is at (0.236, +0.31) in base_link
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.20)
#         time.sleep(1.0)

#         # ── Step 4: descend to grasp height
#         # z=0.12 — safer than 0.08, avoids IK failure near table
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.15)
#         time.sleep(1.0)

#         # ── Step 5: close gripper
#         planner.close_gripper()
#         time.sleep(0.5)

#         # ── Step 6: attach in Isaac Sim
#         planner.attach_object()
#         time.sleep(0.5)

#         # ── Step 7: lift straight up (safe height before moving)
#         planner.move_to_xyz(x=0.15, y=0.25, z=0.20)
#         time.sleep(1.0)

#         # ── Step 8: move to bin — bin is at (+x, -y)
#         # TODO: replace with exact bin coordinates from Isaac Sim
#         planner.move_to_xyz(x=0.20, y=-0.25, z=0.12)
#         time.sleep(1.0)

#         # ── Step 9: detach in Isaac Sim
#         planner.detach_object()
#         time.sleep(0.5)

#         # ── Step 10: open gripper
#         planner.open_gripper()
#         time.sleep(1.0)

#         # ── Step 11: return home
#         planner.move_to_named_target('start')

#         planner.get_logger().info("=== TEST COMPLETE ===")

#     except KeyboardInterrupt:
#         pass
#     finally:
#         planner.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()



# # ════════════════════════════════════════════════════════════════════════
# # STANDALONE TEST — run this node directly to verify everything works
# # ════════════════════════════════════════════════════════════════════════
# # def main(args=None):
# #     rclpy.init(args=args)
# #     planner = MotionPlanner()

# #     try:
# #         planner.get_logger().info("=== TEST SEQUENCE STARTING ===")

# #         # Test 1: move to start position
# #         planner.get_logger().info("--- Test 1: Move to 'start' ---")
# #         ok = planner.move_to_named_target('start')
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")
# #         time.sleep(1.0)

# #         # Test 2: open gripper
# #         planner.get_logger().info("--- Test 2: Open gripper ---")
# #         ok = planner.open_gripper()
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")
# #         time.sleep(1.0)

# #         # Test 3: move to Position_1
# #         planner.get_logger().info("--- Test 3: Move to 'Position_1' ---")
# #         ok = planner.move_to_named_target('Position_1')
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")
# #         time.sleep(1.0)

# #         # Test 4: close gripper
# #         planner.get_logger().info("--- Test 4: Close gripper ---")
# #         ok = planner.close_gripper()
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")
# #         time.sleep(1.0)

# #         # Test 5: attach object (Isaac Sim)
# #         planner.get_logger().info("--- Test 5: Attach object ---")
# #         planner.attach_object()
# #         time.sleep(1.0)

# #         # Test 6: move to Position_2
# #         planner.get_logger().info("--- Test 6: Move to 'Position_2' ---")
# #         ok = planner.move_to_named_target('Position_2')
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")
# #         time.sleep(1.0)

# #         # Test 7: detach object
# #         planner.get_logger().info("--- Test 7: Detach object ---")
# #         planner.detach_object()
# #         time.sleep(1.0)

# #         # Test 8: return home
# #         planner.get_logger().info("--- Test 8: Return to 'start' ---")
# #         ok = planner.move_to_named_target('start')
# #         planner.get_logger().info(f"Result: {'OK' if ok else 'FAILED'}")

# #         planner.get_logger().info("=== TEST SEQUENCE COMPLETE ===")

# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         planner.destroy_node()
# #         rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()