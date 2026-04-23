# SO101 Pick-and-Place — Architecture & Launch Instructions

**Author:** Anushka Satav  
**Stack:** ROS 2 Humble · Ubuntu 22.04 · Isaac Sim 5.1.0 · MoveIt 2  
**Assignment:** Fireloop Robotics — ROS 2 Practical Assignment

---

## 1. System Architecture

### Node Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Isaac Sim 5.1.0                             │
│                                                                      │
│  ┌─────────────┐   /camera/rgb          ┌──────────────────────┐    │
│  │  RGB-D      │──/camera/depth────────►│  perception_node     │    │
│  │  Camera     │──/camera/camera_info──►│  (so101_state_machine│    │
│  └─────────────┘                        │                      │    │
│                                         │  HSV detection       │    │
│  ┌─────────────┐   /isaac_joint_states  │  Deprojection        │    │
│  │  OmniGraph  │──────────────────────► │  TF transform        │    │
│  │  ROS Bridge │◄─────────────────────  │                      │    │
│  └─────────────┘   /isaac_joint_command └──────────┬───────────┘    │
│                                                    │/red_cup_pose   │
│  ┌─────────────┐   /isaac_attach_cube              │                │
│  │  OmniGraph  │◄──────────────────────────────────┼───────────┐   │
│  │  FixedJoint │                                   │           │   │
│  └─────────────┘                                   ▼           │   │
└──────────────────────────────────────────────────────────────────────┘
                                              ┌────────────────────┐
         ┌────────────────────────────────────►   bt_node          │
         │   /joint_states                    │   (so101_state_    │
┌────────┴───────────────┐                   │    machine)        │
│  ros2_control +        │◄──/move_action────┤                    │
│  MoveIt 2              │                   │  Sequence(memory): │
│                        │                   │  OpenGripper       │
│  arm_controller        │                   │  Grabbing          │
│  gripper_controller    │                   │  AttachCube        │
│  joint_state_broadcaster│                  │  MoveToBoxPosition │
└────────────────────────┘                   │  DetachCube        │
                                             │  OpenGripper       │
                                             │                    │
                                             │  MotionPlanner     │
                                             │  (embedded)        │
                                             └────────────────────┘
```

### Topic / Action / TF Reference

| Interface | Type | Direction | Purpose |
|-----------|------|-----------|---------|
| `/isaac_joint_states` | `sensor_msgs/JointState` | Isaac → ROS | Joint feedback at 20Hz |
| `/isaac_joint_command` | `trajectory_msgs/JointTrajectory` | ROS → Isaac | Joint position commands |
| `/camera/rgb` | `sensor_msgs/Image` | Isaac → ROS | RGB image 1280×720 |
| `/camera/depth` | `sensor_msgs/Image` | Isaac → ROS | Depth in metres (float32) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Isaac → ROS | Intrinsics (fx=fy=874.15) |
| `/red_cup_pose` | `geometry_msgs/PoseStamped` | perception → BT | Cup pose in base_link |
| `/isaac_attach_cube` | `std_msgs/Bool` | BT → Isaac | Trigger FixedJoint |
| `/move_action` | `moveit_msgs/action/MoveGroup` | planner → MoveIt | Motion planning |
| `/joint_states` | `sensor_msgs/JointState` | ros2_control → all | Current joint positions |
| `/debug/red_mask` | `sensor_msgs/Image` | perception → RViz | HSV mask debug |
| `/debug/annotated` | `sensor_msgs/Image` | perception → RViz | Annotated RGB debug |

**TF Tree:**
```
world
  └── base_link          (virtual joint — SRDF)
        └── camera_link  (static TF publisher, x=0.135 z=0.790 roll=π)
        └── ... arm links ...
              └── gripper_link
                    └── gripper_frame_link  (end effector tip)
```

---

## 2. Package Structure

```
so101_ws/src/
├── so101_description/       # URDF + STL meshes (provided)
├── so101_bringup/           # Launch files + RViz configs (provided)
├── so101_moveit_config/     # MoveIt SRDF, kinematics, controllers (modified)
├── so101_motion_planning/   # MotionPlanner class + launch files (NEW)
│   ├── so101_motion_planning/
│   │   └── motion_planning_node.py
│   └── launch/
│       ├── motion_planning_test.launch.py
│       └── pick_and_place_bringup.launch.py
└── so101_state_machine/     # perception_node + bt_node (NEW)
    └── so101_state_machine/
        ├── perception_node.py
        └── bt_node.py
```

---

## 3. Launch Instructions

### Prerequisites

```bash
# Install dependencies
sudo apt install ros-humble-moveit
sudo apt install ros-humble-topic-based-ros2-control
sudo apt install ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
pip install py_trees opencv-python

# Build workspace
cd so101_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

### Isaac Sim Setup (required before any ROS launch)

1. Open Isaac Sim 5.1.0
2. **File → Open** → `fireloop_assignment/isaac-usd/isaac_sim_scene/scene.usda`
3. Set up the OmniGraph Script Node path:
   - **Window → Visual Scripting → Action Graph** → select `GraspAttachGraph`
   - Click **Script Node** → in Property panel:
     - `Use Script File`: Checked
     - `Script File Path`: `/home/<user>/anushkasatav/fireloop_assignment_1/fireloop_assignment/isaac-usd/omni_graph_script_node_usda/attach_detach_fixed_joint.py`
4. Press **Play ▶**

### Option A — Using Launch Files (Recommended)

```bash
# Terminal 1: Already done — Isaac Sim running with Play pressed

# Terminal 2: Full perception stack (bringup + static TF + perception)
ros2 launch so101_motion_planning pick_and_place_bringup.launch.py

# Terminal 3: Run the Behaviour Tree
ros2 run so101_state_machine bt_node
```

### Option B — Manual (5 terminals)

```bash
# Terminal 1: Isaac Sim running with Play pressed

# Terminal 2: MoveIt + ROS 2 Control + RViz
ros2 launch so101_bringup bringup_moveit.launch.py use_fake_hardware:=true

# Terminal 3: Static camera TF
ros2 run tf2_ros static_transform_publisher \
  --x 0.135 --y 0.0 --z 0.790 \
  --roll 3.14159 --pitch 0.0 --yaw 0.0 \
  --frame-id base_link --child-frame-id camera_link

# Terminal 4: Perception node
ros2 run so101_state_machine perception_node

# Terminal 5: Behaviour Tree (full pick-and-place)
ros2 run so101_state_machine bt_node
```

### Motion Planning Test Only (Phase 5 verification)

```bash
# Terminal 1: Isaac Sim running with Play pressed

# Terminal 2: Motion planning test
ros2 launch so101_motion_planning motion_planning_test.launch.py
```

This runs a standalone test sequence: `start → Position_1 → Position_2 → start`
with gripper open/close, verifying MoveIt integration independently.

### Verify System is Running

```bash
# Check controllers are active
ros2 control list_controllers
# Expected:
#   joint_state_broadcaster  active
#   arm_controller           active
#   gripper_controller       active

# Check perception is publishing
ros2 topic echo /red_cup_pose --once

# Check joint states are live
ros2 topic hz /joint_states
# Expected: ~20 Hz

# Check TF tree is complete
ros2 run tf2_ros tf2_echo base_link camera_link
```

---

## 4. Key Design Decisions

**`gripper_frame_link` as end effector:**
The URDF defines `gripper_frame_link` as the actual fingertip frame, ~10cm
below `gripper_link`. Using `gripper_link` caused a systematic ~10cm Z error
in all Cartesian targets. Verified with `tf2_echo base_link gripper_frame_link`.

**Own executor thread in MotionPlanner:**
`rclpy.spin_until_future_complete` cannot be called from a non-main thread
while the node is being spun by `rclpy.spin()`. The planner runs its own
`SingleThreadedExecutor` in a daemon thread, allowing BT background threads
to call any planner method safely via `threading.Event` callbacks.

**Fresh start state on every plan:**
Every `MotionPlanRequest` explicitly sets `start_state` from the latest
`/joint_states` message. Without this, MoveIt uses a stale cached state
which results in incorrect IK configurations and the arm moving to the
wrong position.

**IK disambiguation via shoulder_pan path constraint:**
KDL IK has two symmetric solutions for many targets. A loose `shoulder_pan`
path constraint (±1.8 rad tolerance, weight=0.5) biases the solver toward
the correct side — positive hint for +y targets, negative hint for -y targets.

**`move_to_named_target('start')` for post-grasp transit:**
Direct Cartesian lift from the grasp configuration fails IK (the arm is in
a constrained low-z pose). Using the known-good `start` joint configuration
reliably clears the table and positions the arm for bin transit.

**Non-blocking BT leaves:**
Each leaf starts a daemon thread in `initialise()` and returns `RUNNING`
every 0.1s tick until the thread sets `_status`. This satisfies the
assignment requirement and prevents the BT timer from blocking.

**Isaac attach/detach via OmniGraph FixedJoint:**
Native gripper contact grasping in Isaac Sim requires extensive physics
tuning. The OmniGraph Script Node approach (creating `UsdPhysics.FixedJoint`
programmatically) provides reliable object attachment with a simple Bool
ROS 2 topic interface, allowing focus on the robotics pipeline.

---

## 5. Modified Files

| File | Change |
|------|--------|
| `so101_moveit_config/config/so101_new_calib.ros2_control.xacro` | Plugin → `topic_based_ros2_control/TopicBasedSystem` |
| `so101_moveit_config/config/ros2_controllers.yaml` | Removed `velocity` from state_interfaces |
| `so101_moveit_config/config/joint_limits.yaml` | `max_velocity: 1` → `1.0` (2 places) |
| `~/.bashrc` | Removed `/snap/` from `LD_LIBRARY_PATH` (fixes RViz crash) |
| `isaac-usd/isaac_sim_scene/scene.usda` | Added Camera prim + OmniGraph Camera Helpers |