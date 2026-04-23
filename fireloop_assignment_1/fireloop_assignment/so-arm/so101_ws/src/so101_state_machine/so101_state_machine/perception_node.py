#!/usr/bin/env python3
"""
Red cup perception node — downward-facing camera, Isaac Sim 5.1.0.

Camera setup:
  - Camera at Isaac world (0, 0, 1.800), looking straight down
  - Robot base_link at Isaac world (0, 0, 1.010)
  - Static TF base_link → camera_link:
      --x 0.0 --y 0.0 --z 0.790 --roll 3.14159 --pitch 0.0 --yaw 0.0

Subscribes:
  /camera/rgb         sensor_msgs/Image
  /camera/depth       sensor_msgs/Image
  /camera/camera_info sensor_msgs/CameraInfo

Publishes:
  /red_cup_pose       geometry_msgs/PoseStamped  (in base_link frame)
  /debug/red_mask     sensor_msgs/Image          (binary mask for verification)
  /debug/annotated    sensor_msgs/Image          (RGB with centroid drawn)

Pipeline:
  1. ApproximateTimeSynchronizer on RGB + depth + camera_info
  2. BGR → HSV → red mask (hue 0-10 and 170-180)
  3. Morphological open+close to remove noise
  4. Largest contour → centroid pixel (u, v)
  5. Median depth over 7x7 patch → robust depth value
  6. Pinhole deprojection → PointStamped in camera_link
  7. tf2 transform → base_link
  8. Small calibration correction (measured empirically)
  9. Publish PoseStamped + debug images

Calibration offsets (measured across all 4 table quadrants):
  CALIB_X = -0.017m  (consistent ~1.7cm x overestimate)
  CALIB_Y = -0.031m  (consistent ~3.1cm y overestimate)
  CALIB_Z =  0.000m  (z is cup height above table — acceptable)
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
import tf2_geometry_msgs  # noqa: F401 — registers do_transform_point


# ─────────────────────────────────────────────────────────────
# Calibration — measured empirically across all table quadrants
# Error was consistent within 1cm across all positions tested.
# ─────────────────────────────────────────────────────────────

CALIB_X = -0.037   
CALIB_Y = +0.000   
CALIB_Z =  0.000




class RedCupPerception(Node):

    def __init__(self):
        super().__init__('red_cup_perception')

        # ── Parameters ───────────────────────────────────────────────────
        self.declare_parameter('rgb_topic',        '/camera/rgb')
        self.declare_parameter('depth_topic',      '/camera/depth')
        self.declare_parameter('info_topic',       '/camera/camera_info')
        self.declare_parameter('output_topic',     '/red_cup_pose')
        self.declare_parameter('camera_frame',     'camera_link')
        self.declare_parameter('target_frame',     'base_link')
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

        # ── Subscribers — BEST_EFFORT for Isaac Sim topics ───────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        rgb_sub   = message_filters.Subscriber(
            self, Image, rgb_topic,   qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos)
        info_sub  = message_filters.Subscriber(
            self, CameraInfo, info_topic, qos_profile=sensor_qos)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=10, slop=0.15
        )
        self.ts.registerCallback(self.sync_cb)

        # ── Publishers ───────────────────────────────────────────────────
        self.pose_pub  = self.create_publisher(PoseStamped, output_topic, 10)

        # Debug publishers — view in RViz Image display to verify masking
        self.mask_pub  = self.create_publisher(Image, '/debug/red_mask', 10)
        self.ann_pub   = self.create_publisher(Image, '/debug/annotated', 10)

        self._log_every = 15
        self._count     = 0

        self.get_logger().info(
            f'RedCupPerception started.\n'
            f'  RGB={rgb_topic}  Depth={depth_topic}  Info={info_topic}\n'
            f'  Output={output_topic}\n'
            f'  Debug: /debug/red_mask  /debug/annotated\n'
            f'  Frames: {self.camera_frame} → {self.target_frame}\n'
            f'  Calibration: CALIB_X={CALIB_X}  CALIB_Y={CALIB_Y}'
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

        # ── 2. Red HSV mask ──────────────────────────────────────────────
        # Red wraps around hue=0/180 in HSV so we need two ranges:
        #   Lower red: hue  0-10
        #   Upper red: hue 170-180
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m1   = cv2.inRange(hsv, (0,   100, 50), (10,  255, 255))
        m2   = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        # Morphological cleanup — removes small noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # ── Publish debug mask ───────────────────────────────────────────
        # Add to RViz: Displays → Add → Image → Topic: /debug/red_mask
        # White pixels = detected as red. Verify only red cup is white.
        try:
            self.mask_pub.publish(
                self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            )
        except Exception:
            pass

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

        # ── Publish annotated debug image ────────────────────────────────
        # Add to RViz: Displays → Add → Image → Topic: /debug/annotated
        # Shows RGB image with green circle at detected centroid.
        try:
            ann = bgr.copy()
            cv2.drawContours(ann, [c], -1, (0, 255, 0), 2)
            cv2.circle(ann, (u, v), 8, (0, 255, 0), -1)
            cv2.putText(ann, f'({u},{v})', (u+10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.ann_pub.publish(
                self.bridge.cv2_to_imgmsg(ann, encoding='bgr8')
            )
        except Exception:
            pass

        # ── 4. Depth at centroid — median 7x7 patch ─────────────────────
        # Median over a patch is more robust than a single pixel
        # (handles depth noise and missing values at object edges)
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return

        u0, u1 = max(0, u - 3), min(w, u + 4)
        v0, v1 = max(0, v - 3), min(h, v + 4)
        patch  = depth[v0:v1, u0:u1].astype(np.float32)

        # Isaac Sim publishes depth in metres (float32).
        # Handle uint16 millimetres just in case.
        if depth.dtype in (np.uint16, np.int16):
            patch = patch / 1000.0

        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0.05]   # discard zeros / invalid readings

        if patch.size == 0:
            if log:
                self.get_logger().info(f'No valid depth at ({u},{v}).')
            return

        depth_m = float(np.median(patch))

        # ── 5. Pinhole deprojection → camera_link frame ──────────────────
        # Standard pinhole model:
        #   x_cam = (u - cx) * depth / fx
        #   y_cam = (v - cy) * depth / fy
        #   z_cam = depth
        # TF (roll=π) handles axis remapping to base_link automatically.
        K  = info_msg.k
        fx, fy = K[0], K[4]   # 874.15, 874.15 from Isaac camera_info
        cx, cy = K[2], K[5]   # 640.0,  360.0

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

        # ── 8. Calibration correction ────────────────────────────────────
        # Small empirical correction measured across all 4 table quadrants.
        # Error was consistent (not position-dependent) confirming it is a
        # fixed systematic offset from camera placement approximation.
        pt_base.point.x += CALIB_X
        pt_base.point.y += CALIB_Y
        pt_base.point.z += CALIB_Z

        # ── 9. Publish PoseStamped ───────────────────────────────────────
        pose = PoseStamped()
        pose.header.stamp    = rgb_msg.header.stamp
        pose.header.frame_id = self.target_frame
        pose.pose.position   = pt_base.point
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        if log:
            self.get_logger().info(
                f'Cup pixel=({u},{v})  area={area:.0f}  '
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
