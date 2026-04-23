"""
pick_and_place_bringup.launch.py
================================
Launches the full perception + bringup stack for the SO101 pick-and-place system.

Starts:
  1. SO101 bringup (MoveIt + ROS 2 Control + RViz)
  2. Static TF publisher (base_link → camera_link)
  3. Perception node (RGB-D → /red_cup_pose)

Run separately after this:
  ros2 run so101_state_machine bt_node

Usage:
  ros2 launch so101_motion_planning pick_and_place_bringup.launch.py
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ── Package paths ─────────────────────────────────────────
    bringup_pkg   = get_package_share_directory('so101_bringup')

    # ── 1. SO101 Bringup (MoveIt + ROS 2 Control + RViz) ─────
    so101_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'bringup_moveit.launch.py')
        ),
        launch_arguments={
            'use_fake_hardware': 'true',
        }.items(),
    )

    # ── 2. Static TF: base_link → camera_link ────────────────
    # Camera is at Isaac world (0, 0, 1.800)
    # Robot base is at Isaac world (-0.135, 0, 1.010)
    # x offset: 0.000 - (-0.135) = +0.135m
    # z offset: 1.800 - 1.010   = +0.790m
    # roll=π flips camera Z to point straight down
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf_publisher',
        arguments=[
            '--x', '0.135',
            '--y', '0.0',
            '--z', '0.790',
            '--roll', '3.14159',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
        output='screen',
    )

    # ── 3. Perception Node ────────────────────────────────────
    # Delayed 5s to let MoveIt and ROS 2 Control fully start
    perception_node = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='so101_state_machine',
                executable='perception_node',
                name='red_cup_perception',
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        so101_bringup,
        static_tf,
        perception_node,
    ])