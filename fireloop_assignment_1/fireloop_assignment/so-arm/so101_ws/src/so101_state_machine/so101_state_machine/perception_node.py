
# WORKS WITH THE GRIPPER CONSTRAINT BUT HAS 3 CM ERRORS IN POSE

#!/usr/bin/env python3
"""
Red cup perception node — fixed for downward-facing camera.

Camera setup:
  - Camera is mounted above robot, looking straight down
  - Isaac world position: (0, 0, 1.8)
  - Robot base_link world position: (-0.135, 0, 1.010)
  - Static TF base_link → camera_link:
      translation: (0.135, 0, 0.790)
      rotation:    pitch = +π/2  (camera Z forward → points down in world)

Subscribes: /camera/rgb, /camera/depth, /camera/camera_info
Publishes:  /red_cup_pose (PoseStamped in base_link frame)

Pipeline:
  RGB + depth (approx time-synced)
  → HSV red mask → largest contour → centroid (u,v)
  → median depth in patch
  → deproject via pinhole intrinsics → PointStamped in camera_link
  → TF transform to base_link
  → PoseStamped published on /red_cup_pose
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs  # noqa: F401  — registers do_transform_point


class RedCupPerception(Node):

    def __init__(self):
        super().__init__('red_cup_perception')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('rgb_topic',       '/camera/rgb')
        self.declare_parameter('depth_topic',     '/camera/depth')
        self.declare_parameter('info_topic',      '/camera/camera_info')
        self.declare_parameter('output_topic',    '/red_cup_pose')
        self.declare_parameter('camera_frame',    'camera_link')
        self.declare_parameter('target_frame',    'base_link')
        self.declare_parameter('min_contour_area', 100.0)

        rgb_topic    = self.get_parameter('rgb_topic').value
        depth_topic  = self.get_parameter('depth_topic').value
        info_topic   = self.get_parameter('info_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.min_area     = self.get_parameter('min_contour_area').value

        self.bridge = CvBridge()

        # ── TF ───────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ──────────────────────────────────────────────────
        # Isaac Sim publishes sensor topics with BEST_EFFORT reliability
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        rgb_sub   = message_filters.Subscriber(
            self, Image, rgb_topic, qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos)
        info_sub  = message_filters.Subscriber(
            self, CameraInfo, info_topic, qos_profile=sensor_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.15
        )
        self.ts.registerCallback(self.sync_cb)

        # ── Publisher ────────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, output_topic, 10)

        self._log_every = 15
        self._count     = 0

        self.get_logger().info(
            f'RedCupPerception started.\n'
            f'  RGB={rgb_topic}  Depth={depth_topic}  Info={info_topic}\n'
            f'  Output={output_topic}\n'
            f'  Frames: {self.camera_frame} → {self.target_frame}'
        )

    # ════════════════════════════════════════════════════════════════════
    def sync_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self._count += 1
        log = (self._count % self._log_every == 0)

        # ── 1. Convert images ────────────────────────────────────────────
        try:
            bgr   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        # ── 2. Red mask (two hue ranges) ─────────────────────────────────
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, (0,   100, 50), (10,  255, 255))
        m2  = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── 3. Largest contour → centroid ────────────────────────────────
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if log:
                self.get_logger().info('No red contour detected.')
            return

        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            if log:
                self.get_logger().info(f'Red contour too small: {area:.0f}')
            return

        M = cv2.moments(c)
        if M['m00'] == 0:
            return
        u = int(M['m10'] / M['m00'])
        v = int(M['m01'] / M['m00'])

        # ── 4. Depth at centroid (median patch) ──────────────────────────
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return

        u0, u1 = max(0, u - 3), min(w, u + 4)
        v0, v1 = max(0, v - 3), min(h, v + 4)
        patch = depth[v0:v1, u0:u1].astype(np.float32)

        # Convert mm → m if sensor gives uint16
        if depth.dtype in (np.uint16, np.int16):
            patch = patch / 1000.0

        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0.05]   # drop zeros / invalid

        if patch.size == 0:
            if log:
                self.get_logger().info(f'No valid depth at ({u},{v}).')
            return

        depth_m = float(np.median(patch))

        # ── 5. Pinhole deprojection → camera frame ───────────────────────
        #
        # Standard pinhole model (ROS camera convention):
        #   x_cam = (u - cx) * depth / fx    (right in image  → +x in cam)
        #   y_cam = (v - cy) * depth / fy    (down  in image  → +y in cam)
        #   z_cam = depth                    (into  image     → +z in cam)
        #
        # For a downward-facing camera mounted with pitch=+π/2:
        # The TF handles the rotation automatically — we just deproject
        # using standard pinhole and let tf2 do the frame conversion.
        #
        K  = info_msg.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        x_cam = (u - cx) * depth_m / fx
        y_cam = (v - cy) * depth_m / fy
        z_cam = depth_m

        # ── 6. PointStamped in camera_link ───────────────────────────────
        pt = PointStamped()
        pt.header.stamp    = rgb_msg.header.stamp
        pt.header.frame_id = self.camera_frame
        pt.point.x = x_cam
        pt.point.y = y_cam
        pt.point.z = z_cam

        # ── 7. Transform → base_link ─────────────────────────────────────
        try:
            pt_base = self.tf_buffer.transform(
                pt,
                self.target_frame,
                timeout=rclpy.duration.Duration(seconds=0.3),
            )
        except TransformException as ex:
            if log:
                self.get_logger().warn(f'TF failed: {ex}')
            return
        

        # ── Calibration correction ───────────────────────────────────────
        # Measured offset between perception output and Isaac ground truth.
        # X error: ~+0.117m constant, Y error: ~+0.027m constant
        # CALIB_X = -0.117   # measured correction
        # CALIB_Y = -0.027     # y is close enough
        # CALIB_Z =  0.0     # z offset is cup height, acceptable

        # CALIB_X = 0.0
        # CALIB_Y = 0.0
        # CALIB_Z = 0.0

        # to fix that offset :(
        
        CALIB_X = -0.017
        CALIB_Y = -0.031
        CALIB_Z =  0.0


        pt_base.point.x += CALIB_X
        pt_base.point.y += CALIB_Y
        pt_base.point.z += CALIB_Z

        # ── 8. Publish PoseStamped ───────────────────────────────────────
        pose = PoseStamped()
        pose.header.stamp    = rgb_msg.header.stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position   = pt_base.point
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        if log:
            self.get_logger().info(
                f'Cup pixel=({u},{v})  '
                f'cam=({x_cam:+.3f},{y_cam:+.3f},{z_cam:+.3f})  '
                f'→ {self.target_frame}=('
                f'{pt_base.point.x:+.3f},'
                f'{pt_base.point.y:+.3f},'
                f'{pt_base.point.z:+.3f})'
            )


def main():
    rclpy.init()
    node = RedCupPerception()
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
# Red cup perception node.

# Subscribes: /camera/rgb, /camera/depth, /camera/camera_info
# Publishes:  /red_cup_pose (PoseStamped in base_link frame)

# Pipeline:
#   RGB + depth (approx time-synced) -> HSV red mask -> largest contour
#   -> centroid (u,v) -> depth lookup -> deproject via pinhole intrinsics
#   -> PointStamped in camera_link -> TF transform to base_link
#   -> PoseStamped (position only, identity orientation).
# """

# import numpy as np
# import cv2

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# import message_filters
# from cv_bridge import CvBridge

# from sensor_msgs.msg import Image, CameraInfo
# from geometry_msgs.msg import PoseStamped, PointStamped

# import tf2_ros
# from tf2_ros import TransformException
# # Needed to register do_transform_point for PointStamped
# import tf2_geometry_msgs  # noqa: F401


# class RedCupPerception(Node):
#     def __init__(self):
#         super().__init__('red_cup_perception')

#         # Parameters
#         self.declare_parameter('rgb_topic', '/camera/rgb')
#         self.declare_parameter('depth_topic', '/camera/depth')
#         self.declare_parameter('info_topic', '/camera/camera_info')
#         self.declare_parameter('output_topic', '/red_cup_pose')
#         self.declare_parameter('camera_frame', 'camera_link')
#         self.declare_parameter('target_frame', 'base_link')
#         self.declare_parameter('min_contour_area', 125.0)

#         rgb_topic = self.get_parameter('rgb_topic').value
#         depth_topic = self.get_parameter('depth_topic').value
#         info_topic = self.get_parameter('info_topic').value
#         output_topic = self.get_parameter('output_topic').value
#         self.camera_frame = self.get_parameter('camera_frame').value
#         self.target_frame = self.get_parameter('target_frame').value
#         self.min_area = self.get_parameter('min_contour_area').value

#         self.bridge = CvBridge()

#         # TF
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # Sensor QoS (Isaac publishes with BEST_EFFORT by default)
#         sensor_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=5,
#         )

#         # Subscribers with message_filters for time sync
#         rgb_sub = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=sensor_qos)
#         depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=sensor_qos)
#         info_sub = message_filters.Subscriber(self, CameraInfo, info_topic, qos_profile=sensor_qos)

#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.1
#         )
#         self.ts.registerCallback(self.sync_cb)

#         # Publisher
#         self.pose_pub = self.create_publisher(PoseStamped, output_topic, 10)

#         # Throttle debug logs
#         self._log_every = 15  # print roughly every N frames
#         self._count = 0

#         self.get_logger().info(
#             f'RedCupPerception started. RGB={rgb_topic} Depth={depth_topic} '
#             f'Info={info_topic} Out={output_topic} '
#             f'Frames: {self.camera_frame} -> {self.target_frame}'
#         )

#     def sync_cb(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
#         self._count += 1
#         try:
#             # Convert images
#             bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
#             # Depth in Isaac is usually 32FC1 meters. Handle 16UC1 mm too just in case.
#             depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
#         except Exception as e:
#             self.get_logger().warn(f'cv_bridge conversion failed: {e}')
#             return

#         # Build HSV red mask
#         hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#         m1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
#         m2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
#         mask = cv2.bitwise_or(m1, m2)

#         # Clean up mask (remove noise, close small holes)
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         # Largest contour
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             if self._count % self._log_every == 0:
#                 self.get_logger().info('No red contour detected.')
#             return

#         c = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(c)
#         if area < self.min_area:
#             if self._count % self._log_every == 0:
#                 self.get_logger().info(f'Largest red contour too small: area={area:.0f}')
#             return

#         M = cv2.moments(c)
#         if M['m00'] == 0:
#             return
#         u = int(M['m10'] / M['m00'])
#         v = int(M['m01'] / M['m00'])

#         # Depth at centroid (sample a small patch and take median to be robust)
#         h, w = depth.shape[:2]
#         if not (0 <= u < w and 0 <= v < h):
#             return
#         u0, u1 = max(0, u - 2), min(w, u + 3)
#         v0, v1 = max(0, v - 2), min(h, v + 3)
#         patch = depth[v0:v1, u0:u1].astype(np.float32)

#         # Handle units: if values look like millimeters (ints > 100), convert.
#         if depth.dtype in (np.uint16, np.int16):
#             patch = patch / 1000.0  # mm -> m

#         patch = patch[np.isfinite(patch)]
#         patch = patch[patch > 0.05]  # drop zeros / invalid
#         if patch.size == 0:
#             if self._count % self._log_every == 0:
#                 self.get_logger().info(f'No valid depth at ({u},{v}).')
#             return
#         z = float(np.median(patch))

#         # Pinhole deprojection using K = [fx 0 cx; 0 fy cy; 0 0 1]
#         K = info_msg.k
#         fx, fy = K[0], K[4]
#         cx, cy = K[2], K[5]
#         x_cam = (u - cx) * z / fx
#         y_cam = (v - cy) * z / fy
#         z_cam = z

#         # PointStamped in camera frame
#         pt = PointStamped()
#         pt.header.stamp = rgb_msg.header.stamp
#         pt.header.frame_id = self.camera_frame
#         pt.point.x = x_cam
#         pt.point.y = y_cam
#         pt.point.z = z_cam

#         # Transform to target (base_link)
#         try:
#             pt_base = self.tf_buffer.transform(
#                 pt, self.target_frame, timeout=rclpy.duration.Duration(seconds=0.2)
#             )
#         except TransformException as ex:
#             if self._count % self._log_every == 0:
#                 self.get_logger().warn(f'TF transform failed: {ex}')
#             return

#         # Publish PoseStamped (position only, identity orientation)
#         pose = PoseStamped()
#         pose.header.stamp = rgb_msg.header.stamp
#         pose.header.frame_id = self.target_frame
#         pose.pose.position = pt_base.point
#         pose.pose.orientation.w = 1.0
#         self.pose_pub.publish(pose)

#         if self._count % self._log_every == 0:
#             self.get_logger().info(
#                 f'Cup pixel=({u},{v}) cam=({x_cam:+.3f},{y_cam:+.3f},{z_cam:+.3f}) '
#                 f'-> {self.target_frame}=('
#                 f'{pt_base.point.x:+.3f},{pt_base.point.y:+.3f},{pt_base.point.z:+.3f}) '
#                 f'[GT ~ (0.372,-0.309,0.020)]'
#                 # f'[GT ~ (0.236,-0.310,0.016)]'
#             )
#         # gt_x, gt_y, gt_z = 0.236, -0.31, 0.016
#         # gt_x, gt_y, gt_z = 0.372, -0.309, 0.020


# def main():
#     rclpy.init()
#     node = RedCupPerception()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()