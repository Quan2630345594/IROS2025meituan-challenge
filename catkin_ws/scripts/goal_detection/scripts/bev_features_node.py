#!/usr/bin/env python3
"""Phase 1/3 synchronized ROS geometry diagnostics, no learned predictions."""
import cv2
import numpy as np
import rospy
import message_filters
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs import point_cloud2
from bev.geometry import Grid, correspondence, pillar_statistics, transform


def matrix_from_tf(msg):
    t, q = msg.transform.translation, msg.transform.rotation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = quat/norm
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class FeatureNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.info = None
        self.robot = rospy.get_param('~robot_frame', 'base_link')
        self.grid = Grid(resolution=float(rospy.get_param('~resolution', 0.1)))
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.count_pub = rospy.Publisher('~pillar_log_count', Image, queue_size=1)
        self.height_pub = rospy.Publisher('~pillar_mean_height', Image, queue_size=1)
        self.mask_pub = rospy.Publisher('~camera_observed_mask', Image, queue_size=1)
        self.overlay_pub = rospy.Publisher('~projection_overlay', Image, queue_size=1)
        rospy.Subscriber(rospy.get_param('~camera_info_topic', '/magv/camera/camera_info'), CameraInfo, self.camera_info, queue_size=1)
        image = message_filters.Subscriber(rospy.get_param('~image_topic', '/magv/camera/image_compressed/compressed'), CompressedImage)
        cloud = message_filters.Subscriber(rospy.get_param('~cloud_topic', '/magv/scan/3d'), PointCloud2)
        self.sync = message_filters.ApproximateTimeSynchronizer([image, cloud], 5, float(rospy.get_param('~sync_slop', 0.05)))
        self.sync.registerCallback(self.callback)
        rospy.loginfo('BEV geometry diagnostics only: no trained object predictions')

    def camera_info(self, msg):
        self.info = msg

    def callback(self, image_msg, cloud_msg):
        if self.info is None:
            rospy.logwarn_throttle(5, 'Waiting for CameraInfo')
            return
        try:
            info = self.info
            if info.distortion_model not in ('', 'plumb_bob'):
                raise ValueError('Only plumb_bob calibration is supported')
            if not image_msg.header.frame_id or image_msg.header.frame_id != info.header.frame_id:
                raise ValueError('Image/CameraInfo must use the same optical frame')
            image = self.bridge.compressed_imgmsg_to_cv2(image_msg, 'bgr8')
            if image.shape[:2] != (info.height, info.width):
                raise ValueError('Image dimensions do not match CameraInfo')
            K = np.asarray(info.K).reshape(3, 3)
            if K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError('CameraInfo is not calibrated')
            rectified = cv2.undistort(image, K, np.asarray(info.D), None, K)
            camera_tf = self.buffer.lookup_transform_full(
                info.header.frame_id, image_msg.header.stamp, cloud_msg.header.frame_id,
                cloud_msg.header.stamp, rospy.get_param('~fixed_frame', 'odom'), rospy.Duration(0.1))
            robot_tf = self.buffer.lookup_transform(self.robot, cloud_msg.header.frame_id,
                                                    cloud_msg.header.stamp, rospy.Duration(0.1))
            points = np.array(list(point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)), dtype=np.float32).reshape(-1, 3)
            T_cam, T_robot = matrix_from_tf(camera_tf), matrix_from_tf(robot_tf)
            pillars = pillar_statistics(transform(points, T_robot), self.grid)
            _, uv, rows, cols = correspondence(points, K, T_cam, T_robot, image.shape[:2], self.grid)
            mask = np.zeros(self.grid.shape, np.uint8)
            mask[rows, cols] = 255
            for u, v in uv:
                cv2.circle(rectified, (int(u), int(v)), 1, (0, 255, 0), -1)
            for publisher, data, encoding in [(self.count_pub, pillars[0], '32FC1'),
                    (self.height_pub, pillars[1], '32FC1'), (self.mask_pub, mask, 'mono8')]:
                msg = self.bridge.cv2_to_imgmsg(data, encoding)
                msg.header.stamp = cloud_msg.header.stamp
                msg.header.frame_id = self.robot
                publisher.publish(msg)
            overlay = self.bridge.cv2_to_imgmsg(rectified, 'bgr8')
            overlay.header = image_msg.header
            self.overlay_pub.publish(overlay)
        except (ValueError, cv2.error, tf2_ros.LookupException,
                tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5, 'BEV frame skipped: %s', str(exc))


if __name__ == '__main__':
    rospy.init_node('bev_features_node')
    node = FeatureNode()
    rospy.spin()
