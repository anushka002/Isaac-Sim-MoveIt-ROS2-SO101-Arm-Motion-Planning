#!/usr/bin/env python3
"""
SO101 Behaviour Tree Node
=========================
Implements the full pick-and-place sequence:
  1. OpenGripper
  2. Grabbing  (reads /red_cup_pose → pre-grasp → grasp → close gripper)
  3. AttachCube        ← PROVIDED
  4. MoveToBoxPosition (lift → move above box → descend → place)
  5. DetachCube        ← PROVIDED
  6. OpenGripper

BT Concepts used here:
  - Sequence: runs children in order, stops on first FAILURE
  - Retry(n): retries child up to n times on FAILURE
  - OneShot:  runs the tree once then stops forever

Each leaf uses a background thread for MoveIt calls so update()
never blocks — it just checks a flag and returns RUNNING/SUCCESS/FAILURE.

HOW THREADING WORKS HERE:
  - initialise() is called once when the leaf first becomes active
  - We start a background thread that runs the MoveIt call
  - update() checks self._status every 0.1s tick
  - When thread finishes it sets self._status to SUCCESS or FAILURE
  - update() returns that status → BT moves to next leaf
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
# Scene coordinates (in base_link frame)
# ─────────────────────────────────────────────────────────────

# Box/bin position — from Isaac Sim xformOp:translate
BOX_X = 0.20
BOX_Y = -0.14
BOX_Z = 0.06   # top of box — adjust if needed

# Grasp offset above cup — approach from above
PRE_GRASP_Z_OFFSET = 0.15   # 15cm above cup centroid
GRASP_Z_OFFSET     = 0.06   # 6cm above table (cup rim height)

# Place offsets above box
PRE_PLACE_Z_OFFSET = 0.20   # 20cm above box — safe transit height
PLACE_Z_OFFSET     = 0.10   # 10cm above box — release height

# Fixed attach/detach topic
ATTACH_TOPIC = "/isaac_attach_cube"   # NOTE: no /robot/ prefix
ATTACH_DELAY = 0.5


# ═════════════════════════════════════════════════════════════
# LEAF 1 & 6: OpenGripper
# ═════════════════════════════════════════════════════════════
class OpenGripper(py_trees.behaviour.Behaviour):
    """
    Opens the gripper using MoveIt.

    State machine:
      idle → thread starts → RUNNING → thread done → SUCCESS/FAILURE
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner
        self._status = None      # None=not started, True=success, False=failed
        self._thread = None

    def initialise(self):
        """Called once when this leaf becomes active."""
        self.node.get_logger().info(f'[{self.name}] initialise — opening gripper')
        self._status = None

        # Run MoveIt call in background thread so update() never blocks
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Background thread — does the actual MoveIt call."""
        ok = self.planner.open_gripper()
        self._status = ok   # True or False

    def update(self) -> py_trees.common.Status:
        """Called every 0.1s tick."""
        if self._status is None:
            return py_trees.common.Status.RUNNING   # still working

        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(f'[{self.name}] FAILURE')
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Called when leaf exits (success, failure, or interrupted)."""
        pass


# ═════════════════════════════════════════════════════════════
# LEAF 2: Grabbing
# ═════════════════════════════════════════════════════════════
class Grabbing(py_trees.behaviour.Behaviour):
    """
    Full grasp sequence:
      1. Subscribe to /red_cup_pose — wait for a valid pose
      2. Move to pre-grasp (above cup)
      3. Move to grasp (at cup)
      4. Close gripper

    State machine:
      WAITING_FOR_POSE → MOVING → SUCCESS/FAILURE
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner

        self._cup_pose  = None    # latest PoseStamped from /red_cup_pose
        self._status    = None    # None=running, True=success, False=failed
        self._thread    = None
        self._started   = False

        # Subscribe to perception node output
        self._sub = self.node.create_subscription(
            PoseStamped,
            '/red_cup_pose',
            self._pose_cb,
            10
        )

    def _pose_cb(self, msg: PoseStamped):
        """Store latest cup pose — called from ROS spin thread."""
        self._cup_pose = msg

    def initialise(self):
        """Called once when this leaf becomes active."""
        self.node.get_logger().info(f'[{self.name}] initialise — waiting for cup pose')
        self._status  = None
        self._started = False
        # Keep existing subscription — don't recreate

    def _run(self, cup_pose: PoseStamped):
        """
        Background thread — full grasp sequence.
        cup_pose is a snapshot taken when thread starts.
        """
        cx = cup_pose.pose.position.x
        cy = cup_pose.pose.position.y
        cz = cup_pose.pose.position.z

        self.node.get_logger().info(
            f'[{self.name}] Cup at base_link: '
            f'x={cx:.3f} y={cy:.3f} z={cz:.3f}'
        )

        # Step 1: move to pre-grasp (above cup)
        self.node.get_logger().info(f'[{self.name}] Moving to pre-grasp')
        ok = self.planner.move_to_xyz(cx, cy, cz + PRE_GRASP_Z_OFFSET)
        if not ok:
            self.node.get_logger().error(f'[{self.name}] Pre-grasp FAILED')
            self._status = False
            return

        # Step 2: descend to grasp height
        self.node.get_logger().info(f'[{self.name}] Descending to grasp')
        ok = self.planner.move_to_xyz(cx, cy, cz + GRASP_Z_OFFSET)
        if not ok:
            self.node.get_logger().error(f'[{self.name}] Grasp descent FAILED')
            self._status = False
            return

        # Step 3: close gripper
        self.node.get_logger().info(f'[{self.name}] Closing gripper')
        ok = self.planner.close_gripper()
        if not ok:
            self.node.get_logger().error(f'[{self.name}] Close gripper FAILED')
            self._status = False
            return

        self._status = True

    def update(self) -> py_trees.common.Status:
        """Called every 0.1s tick."""

        # Wait until we have a cup pose
        if self._cup_pose is None:
            self.node.get_logger().info(
                f'[{self.name}] Waiting for /red_cup_pose ...', throttle_duration_sec=2.0
            )
            return py_trees.common.Status.RUNNING

        # Start background thread once we have a pose
        if not self._started:
            self._started = True
            # Take a snapshot of current pose for the thread
            pose_snapshot = self._cup_pose
            self._thread = threading.Thread(
                target=self._run, args=(pose_snapshot,), daemon=True
            )
            self._thread.start()

        # Check thread result
        if self._status is None:
            return py_trees.common.Status.RUNNING

        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(f'[{self.name}] FAILURE')
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass


# ═════════════════════════════════════════════════════════════
# LEAF 4: MoveToBoxPosition
# ═════════════════════════════════════════════════════════════
class MoveToBoxPosition(py_trees.behaviour.Behaviour):
    """
    Move the grasped cup to the box and place it:
      1. Lift straight up (safe transit height)
      2. Move above box
      3. Descend to place height

    Box position is hardcoded from Isaac Sim scene.
    """

    def __init__(self, name: str, node: Node, planner: MotionPlanner):
        super().__init__(name)
        self.node    = node
        self.planner = planner
        self._status = None
        self._thread = None

    def initialise(self):
        """Called once when this leaf becomes active."""
        self.node.get_logger().info(f'[{self.name}] initialise — moving to box')
        self._status = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        """Background thread — full place sequence."""

        # Step 1: lift straight up from grasp position
        self.node.get_logger().info(f'[{self.name}] Lifting up')
        ok = self.planner.move_to_xyz(
            BOX_X, BOX_Y, BOX_Z + PRE_PLACE_Z_OFFSET + 0.10
        )
        # If lift fails try going home first as recovery
        if not ok:
            self.node.get_logger().warn(f'[{self.name}] Lift failed — trying home')
            self.planner.move_to_named_target('start')
            self._status = False
            return

        # Step 2: move above box
        self.node.get_logger().info(f'[{self.name}] Moving above box')
        ok = self.planner.move_to_xyz(BOX_X, BOX_Y, BOX_Z + PRE_PLACE_Z_OFFSET)
        if not ok:
            self.node.get_logger().error(f'[{self.name}] Move above box FAILED')
            self._status = False
            return

        # Step 3: descend to place height
        self.node.get_logger().info(f'[{self.name}] Descending to place')
        ok = self.planner.move_to_xyz(BOX_X, BOX_Y, BOX_Z + PLACE_Z_OFFSET)
        if not ok:
            self.node.get_logger().error(f'[{self.name}] Place descent FAILED')
            self._status = False
            return

        self._status = True

    def update(self) -> py_trees.common.Status:
        """Called every 0.1s tick."""
        if self._status is None:
            return py_trees.common.Status.RUNNING

        if self._status:
            self.node.get_logger().info(f'[{self.name}] SUCCESS')
            return py_trees.common.Status.SUCCESS
        else:
            self.node.get_logger().error(f'[{self.name}] FAILURE')
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        pass


# ═════════════════════════════════════════════════════════════
# PROVIDED: AttachDetachCube (unchanged from template)
# ═════════════════════════════════════════════════════════════
class AttachDetachCube(py_trees.behaviour.Behaviour):
    def __init__(self, name, node, topic_name, attach, delay_sec=1.0):
        super().__init__(name)
        self.node       = node
        self.topic_name = topic_name
        self.attach     = attach
        self.delay_sec  = delay_sec
        self.pub        = self.node.create_publisher(Bool, topic_name, 10)
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
                f'BT: Isaac attach={self.attach} on {self.topic_name}'
            )
            self._done = True
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# ═════════════════════════════════════════════════════════════
# TREE BUILDER
# ═════════════════════════════════════════════════════════════
def create_tree(node: Node, planner: MotionPlanner):
    """
    Build the full behaviour tree.

    Structure:
      OneShot
        └── Sequence (memory=True — remembers which steps completed)
              ├── Retry → OpenGripper
              ├── Retry → Grabbing
              ├── AttachCube          ← PROVIDED
              ├── Retry → MoveToBoxPosition
              ├── DetachCube          ← PROVIDED
              └── Retry → OpenGripper
    """
    STEP_RETRIES = 2

    seq = py_trees.composites.Sequence(name="PickAndPlace", memory=True)

    seq.add_children([
        py_trees.decorators.Retry(
            name="RetryOpenGripper1",
            child=OpenGripper("OpenGripper1", node, planner),
            num_failures=STEP_RETRIES,
        ),
        py_trees.decorators.Retry(
            name="RetryGrabbing",
            child=Grabbing("Grabbing", node, planner),
            num_failures=STEP_RETRIES,
        ),
        # PROVIDED — attaches cup to gripper jaw in Isaac Sim
        AttachDetachCube(
            "AttachCube", node, ATTACH_TOPIC,
            attach=True, delay_sec=ATTACH_DELAY
        ),
        py_trees.decorators.Retry(
            name="RetryMoveToBox",
            child=MoveToBoxPosition("MoveToBoxPosition", node, planner),
            num_failures=STEP_RETRIES,
        ),
        # PROVIDED — releases cup from gripper jaw in Isaac Sim
        AttachDetachCube(
            "DetachCube", node, ATTACH_TOPIC,
            attach=False, delay_sec=ATTACH_DELAY
        ),
        py_trees.decorators.Retry(
            name="RetryOpenGripper2",
            child=OpenGripper("OpenGripper2", node, planner),
            num_failures=STEP_RETRIES,
        ),
    ])

    # OneShot — run the sequence once then stop
    root = py_trees.decorators.OneShot(
        name="RunOnce",
        child=seq,
        policy=py_trees.common.OneShotPolicy.ON_COMPLETION,
    )

    return root


# ═════════════════════════════════════════════════════════════
# BT NODE
# ═════════════════════════════════════════════════════════════
class BTNode(Node):
    """
    Main ROS 2 node that owns the BehaviourTree and ticks it.

    The MotionPlanner is created here and passed to all BT leaves
    so they share the same MoveIt action client.
    """

    def __init__(self):
        super().__init__('so101_bt_node')

        # Create motion planner — shared by all BT leaves
        self.planner = MotionPlanner()

        # Build tree
        self.tree = py_trees.trees.BehaviourTree(
            create_tree(self, self.planner)
        )

        # Tick every 0.1 seconds
        self.timer = self.create_timer(0.1, self._tick)

        self.get_logger().info('SO101 BT node started — ticking every 0.1s')

    def _tick(self):
        self.tree.tick()
        # Print tree status every 10 ticks (1 second)
        # Uncomment below for verbose debugging:
        # print(py_trees.display.ascii_tree(self.tree.root))


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




































# #!/usr/bin/env python3
# """
# INTERVIEW TEMPLATE: ROS2 + py_trees Behaviour Tree (Implementation-Agnostic)

# Task sequence:
# 1) open_gripper
# 2) grabbing
# 3) attach (simulation)        <-- PROVIDED (from our side)
# 4) move_to_box_position
# 5) detach (simulation)        <-- PROVIDED (from our side)
# 6) open_gripper

# Instructions:
# - Implement ONLY the TODO logic in OpenGripper / Grabbing / MoveToBoxPosition.
# - You can use ANY ROS approach you want (services/actions/topics/MoveIt/etc.).
# - Each leaf must return proper py_trees status: RUNNING / SUCCESS / FAILURE.
# - Do not block inside update() (no sleep). Use state/futures/timers if needed.
# """

# #!/usr/bin/env python3
# import time
# import rclpy
# from rclpy.node import Node
# import py_trees
# from std_msgs.msg import Bool


# # -------------------------
# # Candidate BT Leaves (blank)
# # -------------------------
# class OpenGripper(py_trees.behaviour.Behaviour):
#     def __init__(self, name: str, node: Node):
#         super().__init__(name)
#         self.node = node
#         # TODO: initialise any clients/publishers/actions you want here
#         # self.some_client = ...

#     def initialise(self):
#         # TODO: reset internal state (futures/flags/timers) if needed
#         pass

#     def update(self) -> py_trees.common.Status:
#         # TODO: implement open gripper
#         # Return RUNNING until done, then SUCCESS/FAILURE
#         return py_trees.common.Status.FAILURE


# class Grabbing(py_trees.behaviour.Behaviour):
#     def __init__(self, name: str, node: Node):
#         super().__init__(name)
#         self.node = node
#         # TODO: initialise any clients/publishers/actions you want here

#     def initialise(self):
#         # TODO: reset internal state
#         pass

#     def update(self) -> py_trees.common.Status:
#         # TODO: implement grabbing logic
#         return py_trees.common.Status.FAILURE


# class MoveToBoxPosition(py_trees.behaviour.Behaviour):
#     def __init__(self, name: str, node: Node):
#         super().__init__(name)
#         self.node = node
#         # TODO: initialise any clients/publishers/actions 
#     def initialise(self):
#         # TODO: reset internal state
#         pass

#     def update(self) -> py_trees.common.Status:
#         # TODO: implement move to box position
#         return py_trees.common.Status.FAILURE


# # -------------------------
# # PROVIDED: Attach / Detach Cube BT Leaf 
# # -------------------------
# class AttachDetachCube(py_trees.behaviour.Behaviour):
#     def __init__(self, name, node, topic_name, attach, delay_sec=1.0):
#         super().__init__(name)
#         self.node = node
#         self.topic_name = topic_name
#         self.attach = attach
#         self.delay_sec = delay_sec

#         self.pub = self.node.create_publisher(Bool, topic_name, 10)
#         self._start_time = None
#         self._done = False

#     def initialise(self):
#         self._start_time = time.monotonic()
#         self._done = False

#     def update(self):
#         if not self._done and (time.monotonic() - self._start_time) >= self.delay_sec:
#             msg = Bool()
#             msg.data = self.attach
#             self.pub.publish(msg)
#             self.node.get_logger().info(
#                 f"BT: Isaac attach={self.attach} on {self.topic_name}"
#             )
#             self._done = True
#             return py_trees.common.Status.SUCCESS

#         return py_trees.common.Status.RUNNING


# # -------------------------
# # Tree
# # -------------------------
# def create_tree(node: Node):
#     STEP_RETRIES = 2  # optional retry per-step (kept simple)
#     ATTACH_TOPIC = "/robot/isaac_attach_cube"
#     ATTACH_DELAY = 0.5

#     seq = py_trees.composites.Sequence(name="TaskSequence", memory=True)

#     seq.add_children([
#         py_trees.decorators.Retry(
#             "RetryOpen1",
#             OpenGripper("OpenGripper1", node),
#             STEP_RETRIES,
#         ),
#         py_trees.decorators.Retry(
#             "RetryGrabbing",
#             Grabbing("Grabbing", node),
#             STEP_RETRIES,
#         ),

#         # PROVIDED
#         AttachDetachCube("AttachCube", node, ATTACH_TOPIC, attach=True, delay_sec=ATTACH_DELAY),

#         py_trees.decorators.Retry(
#             "RetryMoveToBox",
#             MoveToBoxPosition("MoveToBoxPosition", node),
#             STEP_RETRIES,
#         ),

#         # PROVIDED
#         AttachDetachCube("DetachCube", node, ATTACH_TOPIC, attach=False, delay_sec=ATTACH_DELAY),

#         py_trees.decorators.Retry(
#             "RetryOpen2",
#             OpenGripper("OpenGripper2", node),
#             STEP_RETRIES,
#         ),
#     ])

#     # Run once (optional, keeps it from repeating forever)
#     root = py_trees.decorators.OneShot(
#         name="RunOnce",
#         child=seq,
#         policy=py_trees.common.OneShotPolicy.ON_COMPLETION
#     )

#     return root


# # -------------------------
# # Node
# # -------------------------
# class BTNode(Node):
#     def __init__(self):
#         super().__init__("bt_interview_template_node")

#         self.tree = py_trees.trees.BehaviourTree(create_tree(self))
#         self.timer = self.create_timer(0.1, self._tick)

#         self.get_logger().info("Interview BT template node started.")

#     def _tick(self):
#         self.tree.tick()


# def main():
#     rclpy.init()
#     node = BTNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
