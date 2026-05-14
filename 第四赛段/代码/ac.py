import sys
import time
import os
import math
import numpy as np
from threading import Thread, Lock

import rclpy

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, CameraInfo
from cv_bridge import CvBridge

# OpenCV
import cv2 as cv

# LCM Imports 
import lcm
from robot_control_cmd_lcmt import robot_control_cmd_lcmt
from robot_control_response_lcmt import robot_control_response_lcmt


class Robot_Ctrl(object):
    def __init__(self):
        self.rec_thread = Thread(target=self.rec_responce)
        self.send_thread = Thread(target=self.send_publish)
        self.lc_r = lcm.LCM("udpm://239.255.76.67:7670?ttl=255")
        self.lc_s = lcm.LCM("udpm://239.255.76.67:7671?ttl=255")
        self.cmd_msg = robot_control_cmd_lcmt()
        self.rec_msg = robot_control_response_lcmt()
        self.send_lock = Lock()
        self.delay_cnt = 0
        self.mode_ok = 0
        self.gait_ok = 0
        self.runing = 1

    def run(self):
        self.lc_r.subscribe("robot_control_response", self.msg_handler)
        self.send_thread.start()
        self.rec_thread.start()

    def msg_handler(self, channel, data):
        self.rec_msg = robot_control_response_lcmt().decode(data)
        if self.rec_msg.order_process_bar >= 95:
            self.mode_ok = self.rec_msg.mode
            self.gait_ok = self.rec_msg.gait_id
        else:
            self.mode_ok = 0
            self.gait_ok = 0

    def rec_responce(self):
        while self.runing:
            self.lc_r.handle()
            time.sleep(0.002)

    def Wait_finish(self, mode, gait_id):
        count = 0
        while self.runing and count < 2000:  # 约 10s
            if self.mode_ok == mode and self.gait_ok == gait_id:
                return True
            else:
                time.sleep(0.005)
                count += 1

    def send_publish(self):
        while self.runing:
            self.send_lock.acquire()
            # 心跳 100Hz，life_count 不变时也会发送
            if self.delay_cnt > 20:
                self.lc_s.publish("robot_control_cmd", self.cmd_msg.encode())
                self.delay_cnt = 0
            self.delay_cnt += 1
            self.send_lock.release()
            time.sleep(0.005)

    def Send_cmd(self, msg):
        self.send_lock.acquire()
        self.delay_cnt = 50  # 立即触发一次发送
        self.cmd_msg = msg
        self.send_lock.release()

    def quit(self):
        self.runing = 0
        self.rec_thread.join()
        self.send_thread.join()


def get_yaw_from_quaternion(x, y, z, w):
    """
    手写算法：从四元数 (x,y,z,w) 计算 Yaw (偏航角)
    返回: 弧度制 (-pi 到 pi)
    """
    # 公式: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    # 分子部分 (sin_yaw 的变体)
    sin_yaw = 2.0 * (w * z + x * y)
    
    # 分母部分 (cos_yaw 的变体)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    
    # 计算角度
    yaw = math.atan2(sin_yaw, cos_yaw)
    
    return yaw

def get_pitch_from_quaternion(x, y, z, w):
    """
    计算机器狗的前后倾斜角 (Pitch)。
    
    参数:
        x, y, z, w: 四元数分量 (顺序与你的 yaw 函数保持一致)
        
    返回:
        float: 前后倾斜角 (弧度制)
               > 0 : 通常表示低头 (机头向下)
               < 0 : 通常表示抬头 (机头向上)
               (具体正负取决于 IMU 安装方向，如需反转可返回 -pitch)
    """
    
    # 标准公式 (基于 Z-Y-X 旋转顺序，X轴为前方):
    # sin(pitch) = 2 * (w * y - z * x)
    sinp = 2.0 * (w * y - z * x)
    
    # 处理浮点数误差导致的超出 [-1, 1] 范围的情况 (防止 asin 报错)
    if abs(sinp) >= 1.0:
        # 如果超出范围，直接取 +/- 90度 (pi/2)
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
        
    return pitch


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

        fx = 640 / 2.55
        fy = 640 / 2.55
        cx = 320.0
        cy = 240.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.zeros((4, 1), dtype=np.float32)

        new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (640, 480), np.eye(3), balance=0.5
        )
        self.map1, self.map2 = cv.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (640, 480), cv.CV_16SC2
        )

        self.sub_left = self.create_subscription(
            Image, '/fisheye_left_camera/image_raw', self.left_cb, qos
        )
        self.sub_right = self.create_subscription(
            Image, '/fisheye_right_camera/image_raw', self.right_cb, qos
        )

        self.last_left = None
        self.last_right = None
        self.timer = self.create_timer(0.05, self.show)

        self.get_logger().info('Fisheye undistort node started')

    def left_cb(self, msg):
        self.last_left = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def right_cb(self, msg):
        self.last_right = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def show(self):
        if self.last_left is not None:
            left = self.last_left
            left_ud = cv.remap(left, self.map1, self.map2, cv.INTER_LINEAR)
            cv.imshow('Left (undistorted)', left_ud)

        if self.last_right is not None:
            right = self.last_right
            right_ud = cv.remap(right, self.map1, self.map2, cv.INTER_LINEAR)
            cv.imshow('Right (undistorted)', right_ud)

        cv.waitKey(1)


class ImuTestNode(Node):
    def __init__(self):
        super().__init__('qx_detector_node')
        self.bridge = CvBridge()
        self.current_w = 0
        self.current_pitch = 0
        imu_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.topic_name = '/imu'

        # 订阅仿真摄像头发布的图像话题
        self.subscription = self.create_subscription(
            Imu,
            self.topic_name,
            self.imu_callback,
            imu_qos  # <--- 关键：传入自定义的 QoS 配置
        )

        
        # self.get_logger().info(f"IMU topic geting")
    
    def imu_callback(self, msg:Imu):
        # 1. 提取四元数
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # 2. 调用自写函数计算 Yaw
        yaw_rad = get_yaw_from_quaternion(x, y, z, w)
        
        # 3. 更新变量 (这里用 current_yaw 更准确，如果你坚持用 current_w 也可以)
        self.current_yaw = yaw_rad
        self.current_w = yaw_rad
        # 4. (可选) 转换为角度制方便观察
        yaw_deg = math.degrees(yaw_rad)
        # self.current_yaw = yaw_deg

        pitch_rad = get_pitch_from_quaternion(x, y, z, w)
        pitch_deg = math.degrees(pitch_rad)
        self.current_pitch = pitch_deg
        

class LaneFollowerPP(Node):
    def __init__(self):

        super().__init__('linex_detector')
        # 创建匹配 Gazebo 相机的 QoS 配置
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.cmd_w = None
        # 使用自定义 QoS 订阅
        self.img_sub = self.create_subscription(
            Image, '/down_camera/image_raw', self.image_callback, qos_profile)
        
        self.bridge = CvBridge()
        self.camera_info = None
        
        # 统计信息
        self.frame_count = 0
        self.cmd_vel = 0
        self.last_cmd = -1  # 上次发送的指令
        
        self.get_logger().info('Line detector node started with default forward enabled.')
        
        self.ROI_Y1 = 140
        self.ROI_Y2 = 180
        self.ROI_X1 = 40
        self.ROI_X2 = 280

        self.YELLOW_LOW = np.array([15, 80, 80])
        self.YELLOW_HIGH = np.array([35, 255, 255])

        self.MIN_LANE_WIDTH = 20

        # ⭐ 前方目标点行
        self.TARGET_Y = 10  # ROI内部坐标，越小看得越远

        # ⭐ 控制参数
        self.K_OFFSET =  0.15   # 底部偏移权重
        self.K_CURVE = 0.100    # 前瞻曲率权重

        # 输出限制
        self.STEER_MIN = -0.32
        self.STEER_MAX = 0.32
        self.steer = 0
        self.steer_h =0
        self.min_line_area = 10
        self.first_yellow_line = 0
        self.last_yellow_line = 0
        self.dir = -1
        self.debug = 1
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        if self.dir == 1:
            s_h, steer,vis,mask_vis = self.process_image(cv_image)
        else :
            s_h, steer,vis,mask_vis = self.process_image_re(cv_image)

        self.steer = -steer
        self.steer_h  = -s_h
        if self.debug == 1 :
            cv.imshow("vis",vis)
            cv.imshow("mash",mask_vis)
            cv.waitKey(1)


    def process_image(self, cv_image):
        vis = cv_image.copy()

        # ===== 1. ROI =====
        roi = cv_image[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.YELLOW_LOW, self.YELLOW_HIGH)
        
        
        mask_x = cv.cvtColor(vis, cv.COLOR_BGR2HSV)
        mask_x = cv.inRange(mask_x, self.YELLOW_LOW, self.YELLOW_HIGH)

        h, w = mask.shape
        h_r,w_r = mask_x.shape

        mid_points = {}

        binary_mask = mask_x > 0

        # 2. 检查每一列是否有非零点
        # any(axis=0) 会返回一个长度为 W 的数组，表示每一列是否存在非零点
        col_has_line = np.any(binary_mask, axis=0)

        # 3. 初始化结果数组，默认设为 -1 (表示该列没检测到线)
        # 形状: (W,)
        first_y_positions = np.full(w_r, -1, dtype=np.int32)
        last_y_positions = np.full(w_r, -1, dtype=np.int32)
        # 4. 只对那些有线条的列计算第一个非零位置
        # np.argmax(binary_mask, axis=0) 会返回每一列第一个 True 的索引
        # 注意：如果某列全为 False，argmax 会错误地返回 0，所以必须配合 col_has_line 使用
        all_first_indices = np.argmax(binary_mask, axis=0)

        # 将计算出的有效位置填入结果数组
        first_y_positions[col_has_line] = all_first_indices[col_has_line]

        # --- 结果使用 ---
        # first_y_positions[x] 就是第 x 列从上往下第一个非零点的 y 坐标
        # 例如：获取中间列的结果
        mid_x = w_r // 2
        if first_y_positions[mid_x] != -1:
            self.get_logger().info(f"中间列检测到的线条高度: {first_y_positions[mid_x]}")
        else:
            self.get_logger().info("中间列未检测到线条")

        self.first_yellow_line = first_y_positions[mid_x] 
        # 如果你想画出来看看效果：
        vis_debug = vis.copy()
        for x in range(0, w_r, 5): # 每隔5列画一个点，避免太密
            y = first_y_positions[x]
            if y != -1:
                # 注意：mask_x 是 ROI 还是全图？
                # 如果 mask_x 是全图尺寸，直接用 y。
                # 如果 mask_x 是 ROI (如你代码中的 mask_x 似乎是全图尺寸，因为用了 vis 转换)，直接用。
                # 如果你的 mask_x 实际上是基于 roi 计算的但变量名混淆了，请注意坐标偏移！
                
                # 假设 mask_x 对应的是 vis 的全图尺寸 (根据你的代码逻辑：mask_x 是由 vis 转换来的)
                cv.circle(vis_debug, (x, y), 2, (0, 255, 0), -1)


        col_has_line = np.any(binary_mask, axis=0)

        # 2. 初始化结果数组，默认 -1
        last_y_positions = np.full(w_r, -1, dtype=np.int32)

        # 3. 核心逻辑：找最下面的点
        # 只有当该列确实有线条时才计算，避免全0列的错误
        if np.any(col_has_line):
            # A. 上下翻转掩码 (把最底下的点变成最上面的点)
            # binary_mask 形状: (H, W) -> flipped_mask 形状: (H, W)
            flipped_mask = np.flipud(binary_mask)
            
            # B. 在翻转后的图中找每一列的"第一个"非零点
            # 这对应原图中的"最后一个"非零点
            # indices_from_bottom 是距离底部的距离 (0表示最底行)
            indices_from_bottom = np.argmax(flipped_mask, axis=0)
            
            # C. 还原坐标：y_last = (总高度 - 1) - 距离底部的距离
            # 注意：argmax 返回的是索引，翻转后索引0对应原图索引 H-1
            all_last_indices = (h_r - 1) - indices_from_bottom
            
            # D. 只填充那些确实有线条的列
            last_y_positions[col_has_line] = all_last_indices[col_has_line]

        # --- 结果使用 ---
        mid_x = w_r // 2

        # 获取中间列的最下方黄点
        if last_y_positions[mid_x] != -1:
            self.last_yellow_line = last_y_positions[mid_x]
            self.get_logger().info(f"中间列最下方黄点: Y={self.last_yellow_line}")
        # else:
        #     self.last_yellow_line = -1
        #     self.get_logger().info("中间列未检测到下方黄点")

        self.last_yellow_line = last_y_positions[mid_x]
        # --- 可视化调试 (同时画上最上和最下点) ---
        vis_debug = vis.copy()

        for x in range(0, w_r, 5):
            # 画最上面的点 (绿色)
            y_top = first_y_positions[x] # 你之前计算的变量
            if y_top != -1:
                cv.circle(vis_debug, (x, y_top), 3, (0, 255, 0), -1) # Green
            
            # 画最下面的点 (红色)
            y_bottom = last_y_positions[x]
            if y_bottom != -1:
                cv.circle(vis_debug, (x, y_bottom), 3, (0, 0, 255), -1) # Red
                


        cv.imshow("Debug", vis_debug) # 如果需要显示
        cv.waitKey(1)

        # ===== 2. 行扫描 =====
        for y in range(h):
            row = mask[y]

            state = 0
            left_edge = None
            right_edge = None

            for x in range(w):
                is_yellow = row[x] > 0

                if state == 0:
                    if is_yellow is False and row[x+1] > 0:
                        state = 1
                        left_edge = x
                    elif is_yellow:
                        left_edge = x
                        state = 1


                elif state == 1:
                    if not is_yellow:
                        if row[0] > 0 or x-left_edge<self.min_line_area:
                            state = 0
                        else:
                            state = 2
                            left_edge = x

                elif state == 2:
                    if is_yellow:
                        right_edge = x
                        break

            if left_edge is not None and right_edge is not None:
                if (right_edge - left_edge) > self.MIN_LANE_WIDTH:
                    mid_x = (left_edge + right_edge) // 2
                    mid_points[y] = mid_x
                    # 可视化中点
                    cv.circle(vis,
                               (mid_x + self.ROI_X1, y + self.ROI_Y1),
                               2, (0, 255, 0), -1)

        steering = 0.0
        steer_h = 0.0
        if len(mid_points) > 0:
            # ===== 3. 前方目标点（小y）=====
            target_y = self.TARGET_Y
            if target_y not in mid_points:
                available_y = list(mid_points.keys())
                target_y = min(available_y, key=lambda y: abs(y - self.TARGET_Y))
            x_target = mid_points[target_y]

            # ===== 4. 底部参考点（大y）=====
            bottom_y = max(mid_points.keys())
            x_bottom = mid_points[bottom_y]

            x_center = w // 2
            # 横向偏移（底部）
            offset = x_bottom - x_center

            # 前瞻曲率
            lookahead_offset = x_target - x_bottom
            # curvature = 2 * lookahead_offset / (lookahead_offset**2 + target_y**2 + 1e-6)

            # # ===== 5. 融合控制 =====
            # steering = self.K_OFFSET * offset + self.K_CURVE * curvature
            # steering = np.clip(steering, self.STEER_MIN, self.STEER_MAX)
            
            steering = lookahead_offset * self.K_CURVE
            steer_h = offset * self.K_OFFSET
            # ===== 可视化 =====
            # 前瞻点
            cv.circle(vis,
                       (int(x_target) + self.ROI_X1, int(target_y) + self.ROI_Y1),
                       6, (255, 0, 0), -1)
            # 底部点
            cv.circle(vis,
                       (int(x_bottom) + self.ROI_X1, int(bottom_y) + self.ROI_Y1),
                       6, (0, 0, 255), -1)

        # ===== 6. ROI框 =====
        cv.rectangle(vis,
                      (self.ROI_X1, self.ROI_Y1),
                      (self.ROI_X2, self.ROI_Y2),
                      (255, 255, 0), 2)
        # 中心线
        center_x = (self.ROI_X1 + self.ROI_X2) // 2
        cv.line(vis,
                 (center_x, self.ROI_Y1),
                 (center_x, self.ROI_Y2),
                 (255, 255, 255), 1)

        # 显示转向
        cv.putText(vis,
                    f"steer: {steering:.3f}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        mask_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return steer_h,steering, vis, mask_vis


    def process_image_re(self, cv_image):
        vis = cv_image.copy()

        # ===== 1. ROI =====
        roi = cv_image[self.ROI_Y1:self.ROI_Y2, self.ROI_X1:self.ROI_X2]
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.YELLOW_LOW, self.YELLOW_HIGH)
        
        
        mask_x = cv.cvtColor(vis, cv.COLOR_BGR2HSV)
        mask_x = cv.inRange(mask_x, self.YELLOW_LOW, self.YELLOW_HIGH)

        h, w = mask.shape
        h_r,w_r = mask_x.shape

        mid_points = {}

        binary_mask = mask_x > 0

        # 2. 检查每一列是否有非零点
        # any(axis=0) 会返回一个长度为 W 的数组，表示每一列是否存在非零点
        col_has_line = np.any(binary_mask, axis=0)

        # 3. 初始化结果数组，默认设为 -1 (表示该列没检测到线)
        # 形状: (W,)
        first_y_positions = np.full(w_r, -1, dtype=np.int32)
        last_y_positions = np.full(w_r, -1, dtype=np.int32)
        # 4. 只对那些有线条的列计算第一个非零位置
        # np.argmax(binary_mask, axis=0) 会返回每一列第一个 True 的索引
        # 注意：如果某列全为 False，argmax 会错误地返回 0，所以必须配合 col_has_line 使用
        all_first_indices = np.argmax(binary_mask, axis=0)

        # 将计算出的有效位置填入结果数组
        first_y_positions[col_has_line] = all_first_indices[col_has_line]

        # --- 结果使用 ---
        # first_y_positions[x] 就是第 x 列从上往下第一个非零点的 y 坐标
        # 例如：获取中间列的结果
        mid_x = w_r // 2
        if first_y_positions[mid_x] != -1:
            self.get_logger().info(f"中间列检测到的线条高度: {first_y_positions[mid_x]}")
        else:
            self.get_logger().info("中间列未检测到线条")

        self.first_yellow_line = first_y_positions[mid_x] 
        # 如果你想画出来看看效果：
        vis_debug = vis.copy()
        for x in range(0, w_r, 5): # 每隔5列画一个点，避免太密
            y = first_y_positions[x]
            if y != -1:
                # 注意：mask_x 是 ROI 还是全图？
                # 如果 mask_x 是全图尺寸，直接用 y。
                # 如果 mask_x 是 ROI (如你代码中的 mask_x 似乎是全图尺寸，因为用了 vis 转换)，直接用。
                # 如果你的 mask_x 实际上是基于 roi 计算的但变量名混淆了，请注意坐标偏移！
                
                # 假设 mask_x 对应的是 vis 的全图尺寸 (根据你的代码逻辑：mask_x 是由 vis 转换来的)
                cv.circle(vis_debug, (x, y), 2, (0, 255, 0), -1)


        col_has_line = np.any(binary_mask, axis=0)

        # 2. 初始化结果数组，默认 -1
        last_y_positions = np.full(w_r, -1, dtype=np.int32)

        # 3. 核心逻辑：找最下面的点
        # 只有当该列确实有线条时才计算，避免全0列的错误
        if np.any(col_has_line):
            # A. 上下翻转掩码 (把最底下的点变成最上面的点)
            # binary_mask 形状: (H, W) -> flipped_mask 形状: (H, W)
            flipped_mask = np.flipud(binary_mask)
            
            # B. 在翻转后的图中找每一列的"第一个"非零点
            # 这对应原图中的"最后一个"非零点
            # indices_from_bottom 是距离底部的距离 (0表示最底行)
            indices_from_bottom = np.argmax(flipped_mask, axis=0)
            
            # C. 还原坐标：y_last = (总高度 - 1) - 距离底部的距离
            # 注意：argmax 返回的是索引，翻转后索引0对应原图索引 H-1
            all_last_indices = (h_r - 1) - indices_from_bottom
            
            # D. 只填充那些确实有线条的列
            last_y_positions[col_has_line] = all_last_indices[col_has_line]

        # --- 结果使用 ---
        mid_x = w_r // 2

        # 获取中间列的最下方黄点
        if last_y_positions[mid_x] != -1:
            self.last_yellow_line = last_y_positions[mid_x]
            self.get_logger().info(f"中间列最下方黄点: Y={self.last_yellow_line}")
        # else:
        #     self.last_yellow_line = -1
        #     self.get_logger().info("中间列未检测到下方黄点")

        self.last_yellow_line = last_y_positions[mid_x]
        # --- 可视化调试 (同时画上最上和最下点) ---
        vis_debug = vis.copy()

        for x in range(0, w_r, 5):
            # 画最上面的点 (绿色)
            y_top = first_y_positions[x] # 你之前计算的变量
            if y_top != -1:
                cv.circle(vis_debug, (x, y_top), 3, (0, 255, 0), -1) # Green
            
            # 画最下面的点 (红色)
            y_bottom = last_y_positions[x]
            if y_bottom != -1:
                cv.circle(vis_debug, (x, y_bottom), 3, (0, 0, 255), -1) # Red
                


        cv.imshow("Debug", vis_debug) # 如果需要显示
        cv.waitKey(1)

        # ===== 2. 行扫描 =====
        for y in range(h):
            row = mask[y]

            state = 0
            left_edge = None
            right_edge = None

            for x in range(w - 1, -1, -1):
                is_yellow = row[x] > 0

                if state == 0:
                    if is_yellow is False and row[0] > 0:
                        state = 1
                        right_edge = x
                    elif is_yellow:
                        right_edge = x
                        state = 1


                elif state == 1:
                    if not is_yellow:
                        if row[w-1] > 0 or right_edge-x<self.min_line_area:
                            state = 0
                        else:
                            state = 2
                            right_edge = x

                elif state == 2:
                    if is_yellow:
                        left_edge = x
                        break

            if left_edge is not None and right_edge is not None:
                if (right_edge - left_edge) > self.MIN_LANE_WIDTH:
                    mid_x = (left_edge + right_edge) // 2
                    mid_points[y] = mid_x
                    # 可视化中点
                    cv.circle(vis,
                               (mid_x + self.ROI_X1, y + self.ROI_Y1),
                               2, (0, 255, 0), -1)

        steering = 0.0
        steer_h = 0.0
        if len(mid_points) > 0:
            # ===== 3. 前方目标点（小y）=====
            target_y = self.TARGET_Y
            if target_y not in mid_points:
                available_y = list(mid_points.keys())
                target_y = min(available_y, key=lambda y: abs(y - self.TARGET_Y))
            x_target = mid_points[target_y]

            # ===== 4. 底部参考点（大y）=====
            bottom_y = max(mid_points.keys())
            x_bottom = mid_points[bottom_y]

            x_center = w // 2
            # 横向偏移（底部）
            offset = x_bottom - x_center

            # 前瞻曲率
            lookahead_offset = x_target - x_bottom
            curvature = 2 * lookahead_offset / (lookahead_offset**2 + target_y**2 + 1e-6)

            # ===== 5. 融合控制 =====
            steering = self.K_OFFSET * offset + self.K_CURVE * curvature
            steering = np.clip(steering, self.STEER_MIN, self.STEER_MAX)
            
            steering = lookahead_offset * self.K_CURVE
            steer_h = offset * self.K_OFFSET
            # ===== 可视化 =====
            # 前瞻点
            cv.circle(vis,
                       (int(x_target) + self.ROI_X1, int(target_y) + self.ROI_Y1),
                       6, (255, 0, 0), -1)
            # 底部点
            cv.circle(vis,
                       (int(x_bottom) + self.ROI_X1, int(bottom_y) + self.ROI_Y1),
                       6, (0, 0, 255), -1)

        # ===== 6. ROI框 =====
        cv.rectangle(vis,
                      (self.ROI_X1, self.ROI_Y1),
                      (self.ROI_X2, self.ROI_Y2),
                      (255, 255, 0), 2)
        # 中心线
        center_x = (self.ROI_X1 + self.ROI_X2) // 2
        cv.line(vis,
                 (center_x, self.ROI_Y1),
                 (center_x, self.ROI_Y2),
                 (255, 255, 255), 1)

        # 显示转向
        cv.putText(vis,
                    f"steer: {steering:.3f}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

        mask_vis = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        return steer_h,steering, vis, mask_vis


def main(args=None):
    rclpy.init(args=args)
    Ctrl = Robot_Ctrl()
    Ctrl.run()
    msg = robot_control_cmd_lcmt()


    nodeI = ImuTestNode()
    nodeLine = LaneFollowerPP()
    nodeFish = FisheyeUndistortNode()

    executor = SingleThreadedExecutor()
    executor.add_node(nodeI)
    executor.add_node(nodeLine)
    executor.add_node(nodeFish)


    PHASE_TURN_LEFT_1 = 9    # 初始阶段
    APHASE_SHIBAN = 117
    FIX_DIR = 999

    current_phase = PHASE_TURN_LEFT_1  # 开始

    try:
        msg.mode = 12 # Recovery stand
        msg.gait_id = 0
        msg.life_count += 1 # Command will take effect when life_count update
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(12, 0)
        print("stand ok")
        
        TARGET_W = -1.54        # 目标角度：(0.710 + 0.707) / 2
        Kp = 2.0                # 比例系数：决定响应速度。
                                # 建议从 1.0 开始试，如果太慢加大，如果震荡减小。
        MAX_VEL = 0.32           # 最大角速度：限制远距离时的最高转速 (原固定值为 0.1)
        MIN_VEL_THRESHOLD = 0.6 # 死区阈值：当误差小于此值时，认为已对准
        STOP_VEL = 0.0          # 停止时的速度

        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.3)

                if current_phase == FIX_DIR:
                    Kp = 2             # 比例系数：决定响应速度。
                                            # 建议从 1.0 开始试，如果太慢加大，如果震荡减小。
                    MAX_VEL = 0.52           # 最大角速度：限制远距离时的最高转速 (原固定值为 0.1)
                    STOP_VEL = 0.0  

                    # TARGET_W = 0 shiban 
                    TARGET_W = math.pi/2
                    MIN_VEL_THRESHOLD = 0.05

                    error = nodeI.current_w - TARGET_W
                    while error > math.pi:
                            error -= 2 * math.pi
                    while error < -math.pi:
                            error += 2 * math.pi

                    if abs(error) < MIN_VEL_THRESHOLD:
                        print(f"已对准！误差：{error:.4f} < {MIN_VEL_THRESHOLD}")
                        TARGET_W = 0  # 目标角度：(0.710 + 0.707) / 2
                        Kp = 1.05             # 比例系数：决定响应速度。
                                                # 建议从 1.0 开始试，如果太慢加大，如果震荡减小。
                        MAX_VEL = 0.32           # 最大角速度：限制远距离时的最高转速 (原固定值为 0.1)
                        MIN_VEL_THRESHOLD = 0.10 
                        STOP_VEL = 0.0          

                        error = -nodeLine.steer_h - TARGET_W
                        
                        if abs(error) < MIN_VEL_THRESHOLD:
                            print(f"shuiping已对准！误差：{error:.4f} < {MIN_VEL_THRESHOLD}")
                            current_phase = APHASE_SHIBAN

                        else:
                            vel_z_raw = -Kp * error
                            vel_z = max(-MAX_VEL, min(vel_z_raw, MAX_VEL))
                            # 5. 限幅处理 (Clamping)
                            # 限制最大速度，防止转得太快失控
                            msg.mode = 11
                            msg.gait_id = 26
                            # zuo 
                            msg.rpy_des = [0.0, -0.15, 0.0]  # 抬头
                            msg.vel_des = [0.0, vel_z, 0.0]  # <--- 动态速度在这里
                            msg.life_count = (msg.life_count + 1) % 128
                            
                            Ctrl.Send_cmd(msg)
                            time.sleep(0.02)
                            continue


                    else:
                        vel_z_raw = -Kp * error
                        vel_z = max(-MAX_VEL, min(MAX_VEL, vel_z_raw))
                        
                        # 6. 构建并发送消息
                        msg.mode = 11
                        msg.gait_id = 26
                        msg.rpy_des = [0.0, 0.00, 0.0]  # 抬头
                        msg.vel_des = [0.0, 0.0, vel_z]  # <--- 动态速度在这里
                        msg.life_count = (msg.life_count + 1) % 128
                        
                        Ctrl.Send_cmd(msg)
                        
                        # 调试打印
                        print(f"误差：{error:.4f} | 计算速度：{vel_z:.3f} (Max:{MAX_VEL})")
                        
                        # 7. 保持原有节奏
                        time.sleep(0.2)

                if current_phase == APHASE_SHIBAN:

                    MODE_WALK = 11              # 表格编号 125/303 等: LOCOMOTION (运动控制)
                    GAIT_SLOW_WALK = 27         # 表格编号 303: TROT_SLOW_WI (慢走)
                                                # 优势：专为低速设计，抗干扰能力强
                    
                    WALK_SPEED = 0.2           # 速度 m/s (慢走模式推荐 0.1 ~ 0.3)
                    TOTAL_DISTANCE = 15.0        # 总距离 m
                    STEP_HEIGHT = 0.14          # 抬脚高度 m (表格范围 0~0.06，设 0.08 可触发最大抬腿，足够跨石板)
                    
                    msg.mode = MODE_WALK
                    msg.gait_id = GAIT_SLOW_WALK
                    msg.vel_des = [WALK_SPEED, 0.0, 0.0]  # X 方向速度
                    msg.duration = 0                      # 0 表示持续运行直到速度改变
                    msg.step_height = [STEP_HEIGHT, STEP_HEIGHT] # 前后腿抬脚高度
                    msg.life_count = (msg.life_count + 1) % 128
                    
                    Ctrl.Send_cmd(msg)
                    time.sleep(0.03)

                

        except KeyboardInterrupt:
            pass
        finally:
            executor.shutdown()
            nodeI.destroy_node()
            nodeFish.destroy_node()
            cv.destroyAllWindows()
        
        
        msg.mode = 7    # PureDamper
        msg.gait_id = 0
        msg.life_count += 1
        Ctrl.Send_cmd(msg)
        Ctrl.Wait_finish(7, 0)

    except KeyboardInterrupt:
        pass
    Ctrl.quit()
    sys.exit()

if __name__ == '__main__':
    main()