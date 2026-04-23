"""
motion_planning_test.launch.py
==============================
Launches the motion planning standalone test for the SO101 arm.
This covers Phase 5 of the assignment — verifying MoveIt integration.

Starts:
  1. SO101 bringup (MoveIt + ROS 2 Control + RViz)
  2. Motion planning test node (runs full test sequence automatically)

This is the first-half deliverable:
  - Robot loads in Isaac Sim
  - MoveIt plans and executes trajectories
  - Arm moves between predefined positions
  - Gripper opens and closes
  - Attach/detach signals verified

Prerequisites:
  - Isaac Sim running with scene.usda open and Play pressed

Usage:
  ros2 launch so101_motion_planning motion_planning_test.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ── Package paths ─────────────────────────────────────────
    bringup_pkg = get_package_share_directory('so101_bringup')

    # ── 1. SO101 Bringup (MoveIt + ROS 2 Control + RViz) ─────
    so101_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'bringup_moveit.launch.py')
        ),
        launch_arguments={
            'use_fake_hardware': 'true',
        }.items(),
    )

    # ── 2. Motion Planning Test Node ──────────────────────────
    # Delayed 6s to let MoveIt and controllers fully initialise
    # before sending any motion goals.
    motion_planning_test = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='so101_motion_planning',
                executable='motion_planning_node',
                name='so101_motion_planner',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        so101_bringup,
        motion_planning_test,
    ])