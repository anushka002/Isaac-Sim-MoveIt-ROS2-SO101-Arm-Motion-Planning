#!/usr/bin/env python3
"""
SO101 Behaviour Tree Node
=========================
Implements the full pick-and-place sequence for the SO101 robotic arm.

Based on the provided template structure (OpenGripper / Grabbing /
MoveToBoxPosition + provided AttachDetachCube leaves).

Sequence:
  1.  OpenGripper         → open gripper before moving
  2.  Grabbing            → read /red_cup_pose, move above cup,
                            descend, close gripper
  3.  AttachCube          → Isaac FixedJoint ON  [PROVIDED]
  4.  MoveToBoxPosition   → lift, transit to bin, descend, place
  5.  DetachCube          → Isaac FixedJoint OFF [PROVIDED]
  6.  OpenGripper         → release cup

BT concepts used:
  Sequence(memory=True)  — runs children in order, stops on FAILURE,
                           remembers completed steps on retry
  Retry(n)               — retries child up to n times on FAILURE
  OneShot                — runs entire tree once then stops forever

Each leaf uses a background thread for MoveIt calls so update()
never blocks — it returns RUNNING until the thread sets _status.
"""

import time
import threading

import rclpy
from rclpy.node import Node
import py_trees
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

from so101_motion_planning.motion_planning_node import MotionPlanner


# ─────────────────────────────────────────────────────────────
# Scene coordinates — all in base_link frame
# Tune these based on your Isaac Sim scene layout.
# ─────────────────────────────────────────────────────────────

# Bin/box position from Isaac Sim xformOp:translate
BIN_X = 0.2
BIN_Y = -0.15

# Grasp approach heights (added to perceived cup z)
Z_PRE_GRASP  = 0.22   # safe approach — above cup with gripper closed
Z_GRASP      = 0.15   # descent height — gripper surrounds cup rim

# Transit and placement heights (absolute in base_link)
Z_LIFT       = 0.350   # straight up after grasp — clears table and bin
Z_TRANSIT    = 0.350   # horizontal transit height to bin
Z_ABOVE_BIN  = 0.25   # above bin before descent
Z_PLACE      = 0.2   # release height inside bin

# Attach/detach
ATTACH_TOPIC = "/isaac_attach_cube"
ATTACH_DELAY = 0.5    # seconds to wait before publishing


# ─────────────────────────────────────────────────────────────
# LEAF 1 & 6: OpenGripper
# ─────────────────────────────────────────────────────────────
class OpenGripper(py_trees.behaviour.Behaviour):
    """
    Opens the gripper using MoveIt.
    Called at start (before approach) and at end (after place).

    State machine:
      initialise() → start thread → update() returns RUNNING
      thread done  → _status set  → update() returns SUCCESS/FAILURE
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner
        self._status = None
        self._thread = None

    def initialise(self):
        self.node.get_logger().info(f'[{self.name}] opening gripper')
        self._status = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self._status = self.planner.open_gripper()

    def update(self) -> py_trees.common.Status:
        if self._status is None:
            return py_trees.common.Status.RUNNING
        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS ✓')
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().error(f'[{self.name}] FAILURE ✗')
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass


# ─────────────────────────────────────────────────────────────
# LEAF 2: Grabbing
# ─────────────────────────────────────────────────────────────
class Grabbing(py_trees.behaviour.Behaviour):
    """
    Full grasp sequence:
      1. Wait for /red_cup_pose from perception node
      2. Move to home (safe start)
      3. Move above cup (gripper closed, safe transit)
      4. Open gripper (when directly above cup)
      5. Descend to cup rim height
      6. Close gripper (grip the cup)

    State machine:
      WAITING_FOR_POSE → thread started → RUNNING → SUCCESS/FAILURE
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner

        self._cup_pose = None   # latest PoseStamped from /red_cup_pose
        self._status   = None
        self._thread   = None
        self._started  = False

        # Subscribe to perception output
        self._sub = self.node.create_subscription(
            PoseStamped, '/red_cup_pose', self._pose_cb, 10
        )

    def _pose_cb(self, msg: PoseStamped):
        self._cup_pose = msg

    def initialise(self):
        self.node.get_logger().info(f'[{self.name}] waiting for cup pose')
        self._status  = None
        self._started = False

    def _run(self, cup_pose: PoseStamped):
        cx = cup_pose.pose.position.x
        cy = cup_pose.pose.position.y
        cz = cup_pose.pose.position.z

        self.node.get_logger().info(
            f'[{self.name}] cup at base_link: '
            f'x={cx:.3f} y={cy:.3f} z={cz:.3f}'
        )

        # 1. Safe home first
        self.node.get_logger().info(f'[{self.name}] moving to home')
        if not self.planner.move_to_named_target('start'):
            self._status = False; return

        # 2. Close gripper before moving over scene
        self.node.get_logger().info(f'[{self.name}] closing gripper for transit')
        if not self.planner.close_gripper():
            self._status = False; return

        # 3. Move above cup (safe height, gripper closed)
        pre_z = max(cz + Z_PRE_GRASP, 0.22)
        self.node.get_logger().info(f'[{self.name}] moving above cup z={pre_z:.3f}')
        if not self.planner.move_to_xyz(cx, cy, pre_z):
            self._status = False; return

        # 4. Open gripper when directly above cup
        self.node.get_logger().info(f'[{self.name}] opening gripper above cup')
        if not self.planner.open_gripper():
            self._status = False; return

        # 5. Descend to grasp height
        grasp_z = max(cz + Z_GRASP, 0.08)
        self.node.get_logger().info(f'[{self.name}] descending to z={grasp_z:.3f}')
        if not self.planner.move_to_xyz(cx, cy, grasp_z):
            self._status = False; return

        # 6. Close gripper to grip cup
        self.node.get_logger().info(f'[{self.name}] closing gripper on cup')
        if not self.planner.close_gripper():
            self._status = False; return

        self._status = True

    def update(self) -> py_trees.common.Status:
        if self._cup_pose is None:
            self.node.get_logger().info(
                f'[{self.name}] waiting for /red_cup_pose ...',
                throttle_duration_sec=2.0
            )
            return py_trees.common.Status.RUNNING

        if not self._started:
            self._started = True
            snapshot = self._cup_pose   # snapshot current pose
            self._thread = threading.Thread(
                target=self._run, args=(snapshot,), daemon=True
            )
            self._thread.start()

        if self._status is None:
            return py_trees.common.Status.RUNNING
        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS ✓')
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().error(f'[{self.name}] FAILURE ✗')
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass


# ─────────────────────────────────────────────────────────────
# LEAF 4: MoveToBoxPosition
# ─────────────────────────────────────────────────────────────
class MoveToBoxPosition(py_trees.behaviour.Behaviour):
    """
    Carry cup from grasp position to bin and place it:
      1. Lift straight up (stay at cup x,y to avoid sweeping)
      2. Transit horizontally to above bin
      3. Descend into bin
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner
        self._status = None
        self._thread = None

    def initialise(self):
        self.node.get_logger().info(f'[{self.name}] starting place sequence')
        self._status = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # # 1. Lift straight up
        # self.node.get_logger().info(f'[{self.name}] lifting to z={Z_LIFT:.3f}')
        # if not self.planner.move_to_xyz(BIN_X, BIN_Y, Z_LIFT,
        #                                  constrain_orientation=False):
        #     self.node.get_logger().warn(f'[{self.name}] lift failed — attempting recovery')
        #     self.planner.move_to_named_target('start')
        #     self._status = False; return
            
        # Step 1: lift straight UP from current position (cup x,y)
        # Use named target 'start' as intermediate safe position
        self.node.get_logger().info(f'[{self.name}] moving to safe home first')
        if not self.planner.move_to_named_target('start'):
            self._status = False; return

        # 2. Transit to above bin
        self.node.get_logger().info(
            f'[{self.name}] transit to bin x={BIN_X:.3f} y={BIN_Y:.3f} '
            f'z={Z_TRANSIT:.3f}'
        )
        if not self.planner.move_to_xyz(BIN_X, BIN_Y, Z_TRANSIT,
                                         constrain_orientation=False):
            self._status = False; return

        # 3. Descend into bin
        self.node.get_logger().info(f'[{self.name}] descending to z={Z_PLACE:.3f}')
        if not self.planner.move_to_xyz(BIN_X, BIN_Y, Z_PLACE,
                                         constrain_orientation=False):
            self._status = False; return
            # Wait for arm to fully stop before detach
        time.sleep(1.0)   # ← ADD THIS
        self._status = True

    def update(self) -> py_trees.common.Status:
        if self._status is None:
            return py_trees.common.Status.RUNNING
        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS ✓')
            return py_trees.common.Status.SUCCESS
        self.node.get_logger().error(f'[{self.name}] FAILURE ✗')
        return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass


# ─────────────────────────────────────────────────────────────
# PROVIDED: AttachDetachCube (unchanged from template)
# ─────────────────────────────────────────────────────────────
class AttachDetachCube(py_trees.behaviour.Behaviour):
    """
    Publishes a Bool to /isaac_attach_cube after a short delay.
    attach=True  → Isaac OmniGraph creates FixedJoint (cup sticks to jaw)
    attach=False → Isaac OmniGraph removes FixedJoint (cup released)
    """

    def __init__(self, name, node, topic_name, attach, delay_sec=1.0):
        super().__init__(name)
        self.node        = node
        self.topic_name  = topic_name
        self.attach      = attach
        self.delay_sec   = delay_sec
        self.pub         = self.node.create_publisher(Bool, topic_name, 10)
        self._start_time = None
        self._done       = False

    def initialise(self):
        self._start_time = time.monotonic()
        self._done = False

    def update(self):
        if not self._done and \
                (time.monotonic() - self._start_time) >= self.delay_sec:
            msg = Bool()
            msg.data = self.attach
            self.pub.publish(msg)
            self.node.get_logger().info(
                f"BT: Isaac attach={self.attach} → {self.topic_name}"
            )
            self._done = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        pass


# ─────────────────────────────────────────────────────────────
# TREE BUILDER
# ─────────────────────────────────────────────────────────────
def create_tree(node: Node, planner: MotionPlanner):
    """
    Build the full pick-and-place behaviour tree.

    Structure:
      OneShot (run once then stop)
        └── Sequence (memory=True)
              ├── Retry → OpenGripper       ← open before approach
              ├── Retry → Grabbing          ← detect + grasp
              ├── AttachCube                ← PROVIDED
              ├── Retry → MoveToBoxPosition ← carry + place
              ├── DetachCube                ← PROVIDED
              └── Retry → OpenGripper       ← release
    """
    RETRIES = 2

    seq = py_trees.composites.Sequence(name="PickAndPlace", memory=True)

    seq.add_children([
        py_trees.decorators.Retry(
            name="RetryOpenGripper1",
            child=OpenGripper("OpenGripper1", node, planner),
            num_failures=RETRIES,
        ),
        py_trees.decorators.Retry(
            name="RetryGrabbing",
            child=Grabbing("Grabbing", node, planner),
            num_failures=RETRIES,
        ),
        # PROVIDED — attaches cup to gripper jaw in Isaac Sim
        AttachDetachCube(
            "AttachCube", node, ATTACH_TOPIC,
            attach=True, delay_sec=ATTACH_DELAY
        ),
        py_trees.decorators.Retry(
            name="RetryMoveToBox",
            child=MoveToBoxPosition("MoveToBoxPosition", node, planner),
            num_failures=RETRIES,
        ),
        # PROVIDED — releases cup from gripper jaw in Isaac Sim
        AttachDetachCube(
            "DetachCube", node, ATTACH_TOPIC,
            attach=False, delay_sec=ATTACH_DELAY
        ),
        py_trees.decorators.Retry(
            name="RetryOpenGripper2",
            child=OpenGripper("OpenGripper2", node, planner),
            num_failures=RETRIES,
        ),
    ])

    root = py_trees.decorators.OneShot(
        name="RunOnce",
        child=seq,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION,
    )
    return root


# ─────────────────────────────────────────────────────────────
# BT NODE
# ─────────────────────────────────────────────────────────────
class BTNode(Node):
    """
    Main ROS 2 node. Creates the MotionPlanner, builds the BehaviourTree,
    and ticks it every 0.1s via a timer.
    """

    def __init__(self):
        super().__init__('bt_node')

        # Motion planner — shared by all BT leaves
        self.planner = MotionPlanner()

        # Build and display tree structure
        self.tree = py_trees.trees.BehaviourTree(
            create_tree(self, self.planner)
        )
        self.get_logger().info(
            '=== Behaviour Tree starting ===\n' +
            py_trees.display.ascii_tree(self.tree.root)
        )

        # Tick every 0.1s
        self.timer = self.create_timer(0.1, self._tick)

    def _tick(self):
        self.tree.tick()
        # Check if tree has finished
        if self.tree.root.status == py_trees.common.Status.SUCCESS:
            self.get_logger().info('=== Pick-and-place COMPLETE ✓ ===')
            self.timer.cancel()
        elif self.tree.root.status == py_trees.common.Status.FAILURE:
            self.get_logger().error('=== Pick-and-place FAILED ✗ ===')
            self.timer.cancel()


def main():
    rclpy.init()
    node = BTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
