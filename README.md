# SO101 Pick-and-Place — ROS 2 + Isaac Sim + MoveIt 2

**Author:** Anushka Satav  
**Stack:** ROS 2 Humble · Ubuntu 22.04 · Isaac Sim 5.1.0 · MoveIt 2

---

## Prerequisites

```bash
sudo apt install ros-humble-moveit
sudo apt install ros-humble-topic-based-ros2-control
sudo apt install ros-humble-tf2-ros ros-humble-tf2-geometry-msgs
pip install py_trees opencv-python
```

---

## Build

```bash
cd fireloop_assignment/so-arm/so101_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

---

## Isaac Sim Setup (do this first — required before any ROS launch)

1. Open Isaac Sim 5.1.0
2. **File → Open** → `fireloop_assignment/isaac-usd/isaac_sim_scene/scene.usda`
3. Set up the attach/detach Script Node:
   - **Window → Visual Scripting → Action Graph** → select `GraspAttachGraph`
   - Click the **Script Node**
   - In the Property panel on the right:
     - `Use Script File` → ✅ checked
     - `Script File Path` → set to the full path:
       ```
       /home/<your_username>/anushkasatav/fireloop_assignment_1/fireloop_assignment/isaac-usd/omni_graph_script_node_usda/attach_detach_fixed_joint.py
       ```
4. Press **Play ▶** in Isaac Sim

---

## Running the Full Pick-and-Place Pipeline

### Option A — Launch Files (2 terminals)

```bash
# Terminal 1 — Full stack (MoveIt + ROS 2 Control + static TF + perception)
cd so101_ws && source install/setup.bash
ros2 launch so101_motion_planning pick_and_place_bringup.launch.py

# Terminal 2 — Behaviour Tree (run after Terminal 1 is fully started)
source install/setup.bash
ros2 run so101_state_machine bt_node
```

### Option B — Manual (4 terminals)

```bash
# Terminal 1 — MoveIt + ROS 2 Control + RViz
source install/setup.bash
ros2 launch so101_bringup bringup_moveit.launch.py use_fake_hardware:=true

# Terminal 2 — Static camera TF
ros2 run tf2_ros static_transform_publisher \
  --x 0.135 --y 0.0 --z 0.790 \
  --roll 3.14159 --pitch 0.0 --yaw 0.0 \
  --frame-id base_link --child-frame-id camera_link

# Terminal 3 — Perception node (red cup detection)
source install/setup.bash
ros2 run so101_state_machine perception_node

# Terminal 4 — Behaviour Tree
source install/setup.bash
ros2 run so101_state_machine bt_node
```

---

## Verify System is Running

```bash
# Controllers must all show 'active'
ros2 control list_controllers

# Perception must be publishing cup pose
ros2 topic echo /red_cup_pose --once

# Joint states must be live at ~20Hz
ros2 topic hz /joint_states
```

---

## Motion Planning Test (Phase 5 verification only)

Tests MoveIt integration independently — no perception or BT needed.

```bash
# Terminal 1 — Isaac Sim running with Play pressed

# Terminal 2
ros2 launch so101_motion_planning motion_planning_test.launch.py
```

The arm will automatically move: `start → open gripper → Position_1 → close gripper → Position_2 → start`

---

## What the Pipeline Does

```
1. Perception node detects red cup → publishes /red_cup_pose
2. Behaviour Tree starts:
   ├── Opens gripper
   ├── Moves above detected cup position
   ├── Opens gripper above cup
   ├── Descends to cup rim height
   ├── Closes gripper (grasp)
   ├── Isaac OmniGraph creates FixedJoint (cup attached to gripper)
   ├── Returns to home position (carrying cup)
   ├── Moves to bin position
   ├── Descends into bin
   ├── Isaac OmniGraph removes FixedJoint (cup released)
   └── Opens gripper
```

---

## Package Overview

| Package | Contents |
|---------|----------|
| `so101_description` | URDF + STL meshes (provided) |
| `so101_bringup` | Launch files + RViz configs (provided) |
| `so101_moveit_config` | MoveIt SRDF, kinematics, controllers (modified) |
| `so101_motion_planning` | `MotionPlanner` class + launch files (new) |
| `so101_state_machine` | `perception_node` + `bt_node` (new) |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `arm_controller` fails to activate | Check `ros2_controllers.yaml` — remove `velocity` from `state_interfaces` |
| Cup not detected | Verify static TF is running and Isaac Sim is in Play mode |
| IK failure (error 99999) | Cup is outside reachable workspace — reset cup to default position |
| Attach not working | Verify Script Node path is set correctly in GraspAttachGraph |
| RViz crashes on startup | Remove `/snap/` entries from `LD_LIBRARY_PATH` in `~/.bashrc` |
