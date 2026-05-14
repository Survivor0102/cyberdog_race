import numpy as np
import cv2 as cv
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FisheyeUndistortNode(Node):
    def __init__(self):
        super().__init__('fisheye_undistort_debug')

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.bridge = CvBridge()

        # ---------- 鱼眼等效内参 (等距模型, HFOV=2.55rad) ----------
        fx = 640 / 2.55
        fy = 640 / 2.55
        cx = 320.0
        cy = 240.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Gazebo 畸变系数全为 0
        self.D = np.zeros((4, 1), dtype=np.float32)

        # 去畸变映射表 (左右共用，因为内参相同)
        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (640, 480), np.eye(3), balance=0.5
        )
        self.map1, self.map2 = cv.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (640, 480), cv.CV_16SC2
        )

        self.sub_left = self.create_subscription(
            Image, '/fisheye_left_camera/image_raw', self.left_cb, qos
        )
        self.sub_right = self.create_subscription(
            Image, '/fisheye_right_camera/image_raw', self.right_cb, qos
        )

        self.last_left = None
        self.last_right = None
        self.timer = self.create_timer(0.05, self.show)  # 20fps

        self.get_logger().info(
            'Fisheye debug node started. Windows: Left / Right / Undistorted-L / Undistorted-R')

    def left_cb(self, msg):
        self.last_left = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def right_cb(self, msg):
        self.last_right = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def show(self):
        if self.last_left is not None:
            left = self.last_left
            left_ud = cv.remap(left, self.map1, self.map2, cv.INTER_LINEAR)
            cv.imshow('Left (raw)', left)
            cv.imshow('Left (undistorted)', left_ud)

        if self.last_right is not None:
            right = self.last_right
            right_ud = cv.remap(right, self.map1, self.map2, cv.INTER_LINEAR)
            cv.imshow('Right (raw)', right)
            cv.imshow('Right (undistorted)', right_ud)

        cv.waitKey(1)


def main():
    rclpy.init()
    node = FisheyeUndistortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
