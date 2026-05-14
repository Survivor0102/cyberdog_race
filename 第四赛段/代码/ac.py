import sys
import time
import os
import math
import numpy as np
from threading import Thread, Lock
from enum import Enum, auto
from std_msgs.msg import Int32

# ROS 2 Imports
import rclpy

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu, LaserScan,CameraInfo
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

def draw_numbered_lines(img, lines, line_color=(0, 255, 0), point_color=(0, 0, 255), text_color=(255, 255, 0)):
    """
    在图像上绘制所有直线，加粗起点终点，并在线段中间标注编号。
    
    参数:
    img: 输入图像 (会被直接修改)
    lines: HoughLinesP 输出的线条数据
    line_color: 直线颜色 (B, G, R)，默认绿色
    point_color: 端点颜色 (B, G, R)，默认红色
    text_color: 编号文字颜色 (B, G, R)，默认青色
    """
    if lines is None:
        return img

    debug_img = img.copy() # 建议复制一份，以免破坏原图用于后续处理
    # 如果确定不需要保留原图，也可以直接用 img
    
    for idx, l in enumerate(lines):
        x1, y1, x2, y2 = l[0]
        
        # 1. 绘制直线 (粗细为 2)
        cv.line(debug_img, (x1, y1), (x2, y2), line_color, 2)
        
        # 2. 加粗绘制起点和终点 (画实心圆，半径 4)
        cv.circle(debug_img, (x1, y1), 4, point_color, -1)
        cv.circle(debug_img, (x2, y2), 4, point_color, -1)
        
        # 3. 计算线段中点，用于放置编号
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        
        # 4. 绘制编号背景 (可选：加一个小黑底让文字更清晰)
        # font = cv.FONT_HERSHEY_SIMPLEX
        # text_size = cv.getTextSize(str(idx), font, 0.5, 1)[0]
        # cv.rectangle(debug_img, (mid_x - text_size[0]//2, mid_y - text_size[1] - 2), 
        #              (mid_x + text_size[0]//2, mid_y + 2), (0, 0, 0), -1)
        
        # 5. 绘制编号文字
        # 参数：图像，文字，位置，字体，缩放比例，颜色，粗细
        cv.putText(debug_img, str(idx), (mid_x, mid_y), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return debug_img

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


def foward_113(msg, Ctrl,T):
        msg.mode = 11
        msg.gait_id = 3
        msg.vel_des = [0.3, 0.0, 0.0]
        msg.rpy_des = [0.0, -0.15, 0.0]
        msg.step_height = [0.06, 0.06]
        msg.pos_des = [0, 0, 0.28]
        msg.duration = 0
        msg.life_count = (msg.life_count + 1) % 128
            
        Ctrl.Send_cmd(msg)

        time.sleep(T)


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
        

class VisionUtils:
    def __init__(self, node_logger):
        self.bridge = CvBridge()
        self.logger = node_logger
        self.has_display = os.environ.get('DISPLAY') is not None

    def msg_to_cv(self, msg, encoding='bgr8'):
        try:
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
        except Exception as e:
            self.logger.error(f"CvBridge error: {e}")
            return None

    def show(self, window_name, img):
        if self.has_display and img is not None:
            cv.imshow(window_name, img)
            cv.waitKey(1)

    @staticmethod
    def destroy_all():
        if os.environ.get('DISPLAY') is not None:
            cv.destroyAllWindows()


class DynamicVisionNode(Node):
    def __init__(self, node_name, topic_name, callback_func):
        super().__init__(node_name)
        self.vision = VisionUtils(self.get_logger())
        self.topic_name = topic_name
        self.callback_func = callback_func
        
        # 状态管理
        self.is_active = False
        self.subscription = None
        
        # QoS 配置
        self.qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        
        self.get_logger().info(f"{node_name} 初始化完成 (当前状态：未激活)")

    def enable(self):
        """启动订阅，开始检测"""
        if not self.is_active:
            self.subscription = self.create_subscription(
                Image, self.topic_name, self._internal_callback, self.qos
            )
            self.is_active = True
            self.get_logger().info(f">>> [{self.get_name()}] 已激活，开始接收图像")

    def disable(self):
        """停止订阅，释放资源"""
        if self.is_active and self.subscription is not None:
            self.destroy_subscription(self.subscription)
            self.subscription = None
            self.is_active = False
            self.get_logger().info(f"<<< [{self.get_name()}] 已停用，停止接收图像")

    def _internal_callback(self, msg):
        """内部回调：先检查业务逻辑是否需要处理，再调用具体算法"""
        if not self.is_active:
            return
        
        # 调用子类或传入的具体处理函数
        self.callback_func(msg, self.vision,self)


def process_stick_logic(msg, vision_utils, node_instance):
    """
    杆子检测具体逻辑
    :param msg: ROS Image 消息
    :param vision_utils: 视觉工具类实例
    :param node_instance: StickDetectorNode 实例 (用于更新 self.has_stick)
    """
    # 1. 图像转换
    img = vision_utils.msg_to_cv(msg)
    if img is None: 
        if node_instance: node_instance.has_stick = None
        return

    # 2. 预处理 (严格按照你提供的格式)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 20, 80, apertureSize=3)

    # 3. 霍夫变换 (严格按照你提供的格式)
    lines = cv.HoughLinesP(edges,
                        rho=1,
                        theta=np.pi / 180,
                        threshold=20,
                        minLineLength=10,
                        maxLineGap=50)
    
    # 4. 调试可视化 (使用已实现的 draw_numbered_lines)
    # 注意：如果 lines 为 None，draw_numbered_lines 需要能处理，或者这里加个判断
    if lines is not None:
        test_img = draw_numbered_lines(img.copy(), lines)
        cv.imshow("line_img", test_img) 
    else:
        cv.imshow("line_img", img)
    
    cv.waitKey(1)

    # 5. 核心逻辑：寻找垂直杆子
    has_stick = None  # 初始化局部变量
    # h, w = img.shape[:2] # 原代码中有这行，如果后续没用可以注释掉，这里保留以防万一
    
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # 筛选条件：垂直 (dy大, dx小)
            if dy > 50 and dx < 5:
                print("findxxxxxxxxxxxxxx")
                has_stick = ((x1, y1), (x2, y2))
                break # 找到一根典型的即可退出循环

    # 6. 【关键步骤】将结果回写到节点实例
    if node_instance:
        node_instance.has_stick = has_stick
    
    # 7. 最终可视化 (严格按照你提供的格式，修正了 self 引用错误)
    if has_stick is not None:
        color = (0, 0, 255) 
        thickness = 2    
        # 修正：原代码写的是 self.has_stick[0]，但在独立函数中应使用局部变量 has_stick
        cv.line(img, has_stick[0], has_stick[1], color, thickness)
        
        # 建议使用 vision_utils.show 以兼容无显示器环境，若必须用 cv.imshow 也可
        cv.imshow("stick_find", img)
        cv.waitKey(1)
    else:
        # 可选：如果没有找到，是否要关闭窗口或显示原图？
        pass
    

class StickDetectorNode(DynamicVisionNode):
    def __init__(self):
        super().__init__('stick_detector', '/rgb_camera/image_raw', process_stick_logic)
        self.stick_found = False
        self.has_stick = None

class LaserScanNode(Node):
    def __init__(self):
        super().__init__('laser_scan_detector_node')
        
        self.ranges = [] 
        self.min_range = 0.0
        self.front_ranges = [] # 专门存储前方角度的数据
        
        scan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT, # 激光雷达数据量大，通常用 BEST_EFFORT
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # 常见的激光雷达话题名称 (Gazebo 通常是 /scan, 实机可能是 /lidar_points 或 /scan_raw)
        self.topic_name = '/scan'

        self.subscription = self.create_subscription(
            LaserScan,
            self.topic_name,
            self.scan_callback,
            scan_qos
        )
        
        self.get_logger().info(f"LaserScan node started, listening to: {self.topic_name}")

    def scan_callback(self, msg: LaserScan):
        # 1. 原始数据获取
        # msg.ranges 是一个 float32 的列表，长度通常为 360, 720, 1080 等
        raw_ranges = list(msg.ranges)
        
        # 2. 数据清洗：替换 inf (无穷大) 和 nan 为一个超大值 (例如 100.0 米)，防止计算报错
        clean_ranges = []
        for r in raw_ranges:
            if math.isinf(r) or math.isnan(r):
                clean_ranges.append(0) # 视为“无限远”
            else:
                clean_ranges.append(r)
        
        self.ranges = clean_ranges
        
        # 3. 提取感兴趣区域 (ROI) - 例如：只取正前方 ±45 度的数据
        # 计算公式：索引 = (角度 - 最小角度) / 角度增量
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        
        # 假设我们要取 -45度 到 +45度 (即 -0.785 到 0.785 弧度)
        target_angle_min = -0.785 
        target_angle_max = 0.785
        
        start_idx = int((target_angle_min - angle_min) / angle_inc)
        end_idx = int((target_angle_max - angle_min) / angle_inc)
        
        # 边界保护，防止索引越界
        # start_idx = max(0, start_idx)
        # end_idx = min(len(self.ranges), end_idx)
        start_idx  =  0
        end_idx = len(self.ranges)
        if start_idx < end_idx:
            self.front_ranges = self.ranges[start_idx:end_idx]
            
            # 4. 数据分析
            if len(self.front_ranges) > 0:
                # --- 计算最小距离 ---
                self.min_range = max(self.front_ranges)
                
                # --- 【新增】计算平均距离 ---
                # sum() 求和，len() 求个数
                self.avg_range = sum(self.front_ranges) / len(self.front_ranges)
                
                # 打印日志验证
                # self.get_logger().info(
                #     f"Front Stats -> Min: {self.min_range:.2f} m, Avg: {self.avg_range:.2f} m, Count: {len(self.front_ranges)}"
                # )
            else:
                # 理论上不会进这里，因为上面判断了 len > 0
                self.min_range = 100.0
                self.avg_range = 100.0
                
        else:
            self.min_range = 100.0
            self.avg_range = 100.0
            self.front_ranges = []

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

class LineDetectorNode(Node):
    def __init__(self):
        super().__init__('line_detector_node')
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.declare_parameter('hue_low', 20)
        self.declare_parameter('hue_high', 30)
        self.declare_parameter('sat_low', 100)
        self.declare_parameter('val_low', 100)
        self.declare_parameter('turn_threshold', 20)
        self.declare_parameter('roi_height_ratio', 0.4)
        self.declare_parameter('roi_width_ratio', 0.6)
        self.declare_parameter('default_forward', True)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('show_window', True)

        self.cmd_w = None

        self.img_sub = self.create_subscription(
            Image, '/down_camera/image_raw', 
            self.image_callback, qos_profile)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/down_camera/camera_info', 
            self.camera_info_callback, qos_profile)
        
        self.bridge = CvBridge()
        self.camera_info = None

        self.frame_count = 0
        self.cmd_vel = 0
        self.last_cmd = -1  # 上次发送的指令


        self.fx = 177.69738981169561
        self.fy = 177.69738981169561
        self.cx = 160.5
        self.cy = 90.5
        self.img_width = 320
        self.img_height = 180
        
        # --- 2. 机器人物理参数 (来自 URDF) ---
        self.cam_height = 0.29      # 相机离地高度 (米)
        self.cam_pitch = np.radians(30) # 相机俯仰角 (30度)
        
        # --- 3. 可调节的鸟瞰图窗口参数 (米) ---
        # 你想看前方多远到多远的区域？修改这里即可调整“特定窗口”
        self.view_distance_near = 0   # 最近距离 (避开底盘)
        self.view_distance_far = 1.5    # 最远距离 (聚焦赛道)
        self.view_width_total = 1     # 覆盖的总宽度 (米)
        
        # 输出鸟瞰图的分辨率 (像素)
        self.bev_width = 300
        self.bev_height = 400
        
        self.yellow_lower = np.array([15, 40, 30]) 
        self.yellow_upper = np.array([45, 255, 255])
        
        # --- ROI 配置 ---
        self.roi_y_start = 140
        self.roi_y_end = 180
        self.min_line_area = 50  # 稍微调大，过滤噪点
        
        # --- 形态学核 ---
        self.kernel_close = np.ones((5, 5), np.uint8) # 用于连接断裂的黄线
        self.kernel_open = np.ones((3, 3), np.uint8)
        
        # --- 滤波配置 ---
        self.alpha_slope = 0.2
        self.alpha_error = 0.2
        self.deadzone_slope = 0.1
        self.deadzone_error = 5.0
        self.weight_power = 2.0
        
        # --- 状态缓存 ---
        self.last_smooth_slope = 0.0
        self.last_smooth_error = 0.0
        self.frame_count = 0

        # 状态缓存 (初始化为0或None)
        self.last_smooth_slope = 0.0
        self.last_smooth_error = 0.0
        self.frame_count = 0
        self.slope = 0
        self.error = 0

    
    def camera_info_callback(self, msg):
        self.camera_info = msg


    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        
        full_debug, control_data = self.process_image(cv_image)
        self.slope = control_data['slope']
        self.error = control_data['lateral_error']


    def process_image(self, bgr_image):
        height, width = bgr_image.shape[:2]
        center_x = width // 2 - 50


    def update_transform_matrix(self):
        """根据物理参数重新计算透视变换矩阵"""
        src_pts = []
        
        # 定义四个角的物理坐标 (相对于相机正下方的地面点)
        # 顺序：左上(远左), 右上(远右), 右下(近右), 左下(近左)
        # 注意：在图像中，"远"对应较小的 Y (顶部)，"近"对应较大的 Y (底部)
        phys_corners = [
            (-self.view_width_total / 2, self.view_distance_far), # 左上
            (self.view_width_total / 2, self.view_distance_far),  # 右上
            (self.view_width_total / 2, self.view_distance_near), # 右下
            (-self.view_width_total / 2, self.view_distance_near) # 左下
        ]
        
        for w_offset, distance in phys_corners:
            x_img, y_img = self.world_to_pixel(distance, w_offset)
            src_pts.append([x_img, y_img])
            
        self.src_pts = np.float32(src_pts)
        
        # 目标点：矩形鸟瞰图
        self.dst_pts = np.float32([
            [0, 0],
            [self.bev_width, 0],
            [self.bev_width, self.bev_height],
            [0, self.bev_height]
        ])
        
        # 计算矩阵
        self.matrix = cv.getPerspectiveTransform(self.src_pts, self.dst_pts)
        
        # 打印调试信息
        print(f"[BEV] 变换矩阵已更新:")
        print(f"  物理范围: [{self.view_distance_near}m - {self.view_distance_far}m], 宽 {self.view_width_total}m")
        print(f"  图像源点: {self.src_pts}")

    def world_to_pixel(self, distance, width_offset):
        """
        将地面物理坐标 (距离，横向偏移) 映射到图像像素坐标
        基于针孔相机模型和倾斜角度几何推导
        """
        # 1. 计算视线与水平面的夹角 (gamma)
        # tan(gamma) = height / distance
        gamma = np.arctan2(self.cam_height, distance)
        
        # 2. 计算视线与相机光轴的夹角 (alpha)
        # alpha = gamma - pitch (因为 pitch 是向下倾斜的)
        alpha = gamma - self.cam_pitch
        
        # 3. 计算 Y 像素坐标
        # y = cy + fy * tan(alpha)
        y_pix = self.cy + self.fy * np.tan(alpha)
        
        # 4. 计算 X 像素坐标
        # 首先计算该点的斜距 (Slant Range) R = height / sin(gamma)
        R = self.cam_height / np.sin(gamma)
        
        # 计算该距离处的水平视场角比例
        # 地面宽度 w 对应的相机坐标系下的角度 beta ≈ atan(w / R)
        # 更精确的：x = cx + fx * (X_cam / Z_cam)
        # 在倾斜平面投影中，X_cam = width_offset, Z_cam = R * cos(alpha) ? 
        # 简化工程公式：利用相似三角形，在该深度 R 处，像素密度
        fov_h_half = np.arctan2(self.img_width / 2, self.fx)
        ground_width_at_R = 2 * R * np.tan(fov_h_half)
        
        # 防止除以零
        if ground_width_at_R <= 0: ground_width_at_R = 0.001
        
        pixels_per_meter = self.img_width / ground_width_at_R
        x_pix = self.cx + width_offset * pixels_per_meter
        
        return x_pix, y_pix

    def process_image(self, bgr_image):
        """
        主处理函数：集成你的逻辑 + BEV 变换 + 调试窗口
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        height, width = bgr_image.shape[:2]
        
        # 你的原始逻辑 (示例：计算中心偏移)
        center_x = width // 2 - 50
        
        # 确保尺寸匹配内参 (如果不匹配则缩放)
        if height != self.img_height or width != self.img_width:
            bgr_image = cv.resize(bgr_image, (self.img_width, self.img_height))
            height, width = bgr_image.shape[:2]

        # --- 生成调试图层 (可视化源点梯形) ---
        debug_img = bgr_image.copy()
        # 绘制源点四边形
        pts_int = self.src_pts.astype(int)
        cv.polylines(debug_img, [pts_int], True, (0, 255, 0), 2) # 绿色框
        
        # 绘制关键点文字
        for i, pt in enumerate(pts_int):
            label = ["Far-L", "Far-R", "Near-R", "Near-L"][i]
            cv.circle(debug_img, tuple(pt), 5, (0, 0, 255), -1)
            cv.putText(debug_img, label, (pt[0]+5, pt[1]-5), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制用户设定的中心线参考 (你的 center_x 逻辑)
        cv.line(debug_img, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        cv.putText(debug_img, f"Center_X: {center_x}", (center_x + 10, 30), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        # --- 执行透视变换 (生成鸟瞰图) ---
        bev_image = cv.warpPerspective(
            bgr_image, 
            self.matrix, 
            (self.bev_width, self.bev_height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(0, 0, 0) # 黑色填充区域
        )
        
        # --- 显示调试窗口 ---
        cv.imshow("1. Original + ROI Debug", debug_img)
        cv.imshow("2. Bird's Eye View (BEV)", bev_image)
        
        # 按键处理 (按 'q' 退出，按 'u' 更新参数演示)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            return None # 信号退出
        elif key == ord('u'):
            # 演示动态调整：稍微增加观察距离
            self.view_distance_far += 0.1
            self.update_transform_matrix()
            print("[Debug] 参数已更新，下一帧将生效")

        return bev_image

def main(args=None):
    rclpy.init(args=args)
    Ctrl = Robot_Ctrl()
    Ctrl.run()
    msg = robot_control_cmd_lcmt()


    nodeS = StickDetectorNode()
    nodeI = ImuTestNode()
    nodeL = LaserScanNode()
    nodeLine = LaneFollowerPP()

    executor = SingleThreadedExecutor()
    executor.add_node(nodeI)
    executor.add_node(nodeS)
    # executor.add_node(nodeL)
    executor.add_node(nodeLine)


    PHASE_ALIGN = 1      # 阶段 1: 对准 (调整w到0.7085)
    PHASE_FINE_TUNE = 2  # 阶段 2: 精调/结合 nodeS (杆子居中)
    
    PHASE_QR = 0
    XIEHUO = 90
    PHASE_TURN_RIGHT_1 = 3   # 第一次右转90度
    PHASE_FORWARD_1 = 4      # 第一次直走一小段
    PHASE_TURN_RIGHT_2 = 5   # 第二次右转90度
    PHASE_FORWARD_2 = 6      # 第二次直走一段
    PHASE_TURN_AROUND = 7    # 原地转180度
    PHASE_FORWARD_3 = 8      # 第三次直走一段
    PHASE_TURN_LEFT_1 = 9    # 第一次左转90度
    PHASE_FORWARD_4 = 10     # 第四次直走一段（回到起点）
    PHASE_TURN_LEFT_2 = 11   # 第二次左转90度
    PHASE_FORWARD_5 = 12     # 第五次直走一段
    
    PHASE_TURN_RIGHT_3 = 13   # 最后一次右转
    PHASE_FORWARD_6 = 14      # 最后直走一点点
    
    PHASE_S = 19
    TEST = 99

    APHASE_GANZI = 100
    PHASE_ARROW = 101
    APHASE_FORWARD_1 = 102
    APHASE_PO_JIAODU = 103
    APHASE_YELLOW = 104

    APHASE_YELLOW_STOP = 105
    APHASE_YELLOW_FORWARD = 106
    APHASE_QR2_TURN_LEFT = 107
    APHASE_QR2_FORWARD = 108
    APHASE_QR2_TURN_TO = 109
    APHASE_QR2_SCAN = 110
    APHASE_QR2_OVER = 111
    APHASE_FORWARD_KU = 112
    APHASE_QR2_TURN_LEFT_2 = 113
    APHASE_INKU_BACK  = 114
    APHASE_IKU = 115
    APHASE_FORWARD_2 = 116
    APHASE_SHIBAN = 117
    APHASE_TUEN_RIGHT  = 118
    APHASE_FORWARD_ARROW = 119
    APHASE_TUEN_LEFT = 120
    APHASE_S_2 = 121
    FIX_MID = 998
    FIX_DIR = 999

# you 3.14
# hou 1.63
# qian -1.54
# zuo 0
    A_D = 205
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
        nodeS.enable()

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
            nodeS.destroy_node()
        
        
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