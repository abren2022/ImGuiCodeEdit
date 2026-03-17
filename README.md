import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO
import torch



# ==================== 配置参数 ====================
@dataclass
class Config:
    # 视频流配置
    video_source: str = 'rtsp://admin:lsnotes@byd@10.59.77.177:554/Streaming/Channels/3802'

    # ROI区域定义（归一化坐标 [x1,y1,x2,y2] 0-1）
    # 需要根据实际工位布局调整这些ROI
    rois = {
        'workpiece_area': np.array([[220, 238], [391, 246], [398, 349], [202, 345]]),  # 工件区域
        'next_area': np.array([[128, 236], [82, 327], [9, 291], [45, 222]])
    }

    # 检测阈值
    iou_threshold: float = 0.02  # 判断接触的IOU阈值
    dist_threshold: float = 50  # 判断靠近的距离阈值（像素）
    hold_frames: int = 2  # 动作确认需要的连续帧数

    # 显示配置
    window_width: int = 640
    window_height: int = 360
    show_fps: bool = True
    save_video: bool = False


# ==================== 工序定义 ====================
@dataclass
class ProcessStep:
    name: str
    description: str
    check_func: callable  # 判断函数
    timeout: float = 30.0  # 超时时间（秒）


class SOPMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.steps = self._init_steps()
        self.current_step = 0  # 当前进行到的工序索引（0表示未开始）
        self.step_status = [False] * len(self.steps)  # 各工序完成状态
        self.step_start_time = None
        self.frame_count = 0
        self.fps = 0
        self.fps_update_time = time.time()

        # 用于动作确认的计数器
        self.step_confirm_counter = [0] * len(self.steps)

    def _init_steps(self) -> List[ProcessStep]:
        """初始化5道工序"""
        return [
            ProcessStep("TakeWorkpiece", "Pick workpiece from pre station", self._check_take_workpiece),
            ProcessStep("PowerCheck", "Insert plug for testing", self._check_power_on),
            ProcessStep("ScanBarcode", "Scan workpiece barcode", self._check_scan),
            ProcessStep("ApplyLabel", "Attach product label", self._check_label),
            ProcessStep("VisualCheck", "Inspect and transfer to next station", self._check_inspect_and_next)
        ]

    def _check_take_workpiece(self, detections: Dict) -> bool:
        """检查是否完成取工件"""
        if not self._has_object(detections, 'hand') or not self._has_object(detections, 'workpiece'):
            return False

        # 判断手是否在工件区域并接触工件
        hand_boxes = detections.get('hand', [])
        workpiece_boxes = detections.get('workpiece', [])

        for hand_box in hand_boxes:
            for wp_box in workpiece_boxes:
                # 计算IOU判断是否接触
                iou = self._calculate_iou(hand_box, wp_box)
                if iou > self.config.iou_threshold:
                    return True

        return False

    def _check_power_on(self, detections: Dict) -> bool:
        """检查是否完成上电检测（插头插入）"""
        if not all(self._has_object(detections, obj) for obj in ['hand', 'plug', 'workpiece']):
            return False

        hand_boxes = detections.get('hand', [])
        plug_boxes = detections.get('plug', [])
        workpiece_boxes = detections.get('workpiece', [])

        # 判断：手拿插头 + 插头靠近工件
        hand_holding_plug = False
        plug_near_workpiece = False

        for hand_box in hand_boxes:
            for plug_box in plug_boxes:
                #print("hand with plug IOU",self._calculate_iou(hand_box, plug_box))
                if self._calculate_iou(hand_box, plug_box) > self.config.iou_threshold:
                    hand_holding_plug = True
                    break

        for plug_box in plug_boxes:
            for wp_box in workpiece_boxes:
                #print("workpiece with plug IOU", self._calculate_iou(plug_box, wp_box))
                if self._calculate_iou(plug_box, wp_box) > self.config.iou_threshold:
                    plug_near_workpiece = True
                    break

        return plug_near_workpiece or hand_holding_plug

    def _check_scan(self, detections: Dict) -> bool:
        """检查是否完成扫码"""
        if not all(self._has_object(detections, obj) for obj in ['hand', 'scanner', 'workpiece']):
            return False

        hand_boxes = detections.get('hand', [])
        scanner_boxes = detections.get('scanner', [])
        workpiece_boxes = detections.get('workpiece', [])

        # 判断：手拿扫码枪 + 扫码枪对准工件
        hand_holding_scanner = False
        scanner = False

        for hand_box in hand_boxes:
            for scanner_box in scanner_boxes:
                if self._center_distance(hand_box, scanner_box) < self.config.iou_threshold:
                    hand_holding_scanner = True
                    break

        for scanner_box in scanner_boxes:
            for wp_box in workpiece_boxes:
                if self._center_distance(scanner_box, wp_box) < self.config.dist_threshold:
                    scanner = True
                    break

        return hand_holding_scanner or scanner


    def _check_label(self, detections: Dict) -> bool:
        """检查是否完成贴标签"""
        if not all(self._has_object(detections, obj) for obj in ['hand', 'label', 'workpiece']):
            return False

        hand_boxes = detections.get('hand', [])
        label_boxes = detections.get('label', [])

        # 判断：标签接触工件
        for label_box in label_boxes:
            for hd_box in hand_boxes:
                if self._calculate_iou(label_box, hd_box) > self.config.iou_threshold:
                    return True

        return False

    def _check_inspect_and_next(self, detections: Dict) -> bool:
        """检查是否完成目检和流入下工序"""
        if not all(self._has_object(detections, obj) for obj in ['hand', 'workpiece']):
            return False

        hand_boxes = detections.get('hand', [])
        workpiece_boxes = detections.get('workpiece', [])

        # 判断：工件在下工序区域，且手已离开
        workpiece_in_next_area = False
        hand_away = False

        for wp_box in workpiece_boxes:
            if self._is_in_roi(wp_box, 'next_area'):
                workpiece_in_next_area = True
                break

        # 检查手是否靠近工件
        for hand_box in hand_boxes:
            for wp_box in workpiece_boxes:
                if self._calculate_iou(hand_box, wp_box) > self.config.iou_threshold:
                    hand_away = True
                    break

        return workpiece_in_next_area and hand_away

    def _has_object(self, detections: Dict, obj_name: str) -> bool:
        """检查是否存在指定对象的检测结果"""
        return obj_name in detections and len(detections[obj_name]) > 0

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0

    def _center_distance(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框中心点的距离"""
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _is_in_roi(self, box: List[float], roi_name: str) -> bool:
        """判断边界框是否在指定的ROI区域内"""
        if roi_name not in self.config.rois:
            return False

        roi = self.config.rois[roi_name]
        box_center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

        return cv2.pointPolygonTest(roi, box_center, False) >= 0

    def update(self, detections: Dict) -> List[bool]:
        """更新工序状态"""
        self.frame_count += 1

        # 计算FPS
        if time.time() - self.fps_update_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_update_time = time.time()

        # 如果所有工序都已完成，重置状态
        if all(self.step_status):
            if self._check_reset_condition(detections):
                self.current_step = 0
                self.step_status = [False] * len(self.steps)
                self.step_confirm_counter = [0] * len(self.steps)
                self.step_start_time = None

        # 检查当前工序
        if self.current_step < len(self.steps):
            current_step_obj = self.steps[self.current_step]

            # 设置开始时间
            if self.step_start_time is None:
                self.step_start_time = time.time()

            # 检查超时
            if time.time() - self.step_start_time > current_step_obj.timeout:
                print(f"警告：工序 '{current_step_obj.name}' 超时")
                self.step_start_time = time.time()  # 重置计时器

            # 检查当前工序是否完成
            if current_step_obj.check_func(detections):
                self.step_confirm_counter[self.current_step] += 1
                if self.step_confirm_counter[self.current_step] >= self.config.hold_frames:
                    self.step_status[self.current_step] = True
                    print(f"工序 {self.current_step + 1}/{len(self.steps)} 完成: {current_step_obj.name}")
                    self.current_step += 1
                    self.step_start_time = None
            else:
                self.step_confirm_counter[self.current_step] = max(0, self.step_confirm_counter[self.current_step] - 1)

        return self.step_status

    def _check_reset_condition(self, detections: Dict) -> bool:
        """检查是否满足重置条件（新工件开始）"""
        # 简单实现：当检测到新工件在取件区域，且没有手接触时，认为可以重置
        if self._has_object(detections, 'workpiece') and self._has_object(detections, 'hand'):
            workpiece_boxes = detections['workpiece']

            # 检查是否有工件在取件区域且没有手接触
            for wp_box in workpiece_boxes:
                if self._is_in_roi(wp_box, 'workpiece_area'):
                        return True
        return False

    def draw_status(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """在图像上绘制工序状态和检测结果"""
        h, w = frame.shape[:2]

        # 绘制检测框
        colors = {
            'hand': (0, 255, 0),  # 绿色
            'workpiece': (255, 0, 0),  # 蓝色
            'plug': (0, 0, 255),  # 红色
            'scanner': (255, 255, 0),  # 青色
            'label': (255, 0, 255),  # 紫色
        }

        for obj_name, boxes in detections.items():
            if obj_name in colors:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, (box[0], box[1], box[2], box[3]))
                    cv2.rectangle(frame,  (x1, y1), (x2, y2), colors[obj_name], 2)
                    cv2.putText(frame, obj_name, (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[obj_name], 1)

        # 绘制工序进度条
        progress_height = 80
        progress_y = progress_height - 10

        # 背景
        cv2.rectangle(frame, (10, 0),
                      (w - 10, progress_height),
                      (50, 50, 50), -1)

        # 每个工序的进度块
        step_width = (w - 40) // len(self.steps)
        for i, (step, status) in enumerate(zip(self.steps, self.step_status)):
            x = 20 + i * step_width
            y = 0
            width = step_width - 10
            height = 30

            # 颜色：已完成绿色，当前工序黄色，未完成灰色
            if status:
                color = (0, 255, 0)  # 已完成
            elif i == self.current_step and i < len(self.steps):
                color = (0, 255, 255)  # 当前工序
            else:
                color = (100, 100, 100)  # 未完成

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)

            # 工序编号
            cv2.putText(frame, str(i + 1), (x + width // 2 - 5, y + height // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 工序名称
            cv2.putText(frame, step.name, (x, y + height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示当前工序提示
        if self.current_step < len(self.steps):
            current_step = self.steps[self.current_step]
            cv2.putText(frame, f"cur: {current_step.name}",
                        (20, progress_y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 显示FPS
        if self.config.show_fps:
            cv2.putText(frame, f"FPS: {self.fps}",
                        (w - 150, progress_y+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame



# ==================== 模拟检测器 ====================
class MockDetector:
    def __init__(self):

        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        self.models = YOLO("D:/project/Data_Label_Process_Tools/model/best.pt").to(device)
    def detect(self, frame):
            """模拟检测结果"""
            detections = {}
            objs = ['hand', 'scanner','plug','workpiece','label']
            results = self.models(frame, verbose=False)
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    cls = int(box.cls[0])  # 类别索引

                    if box:
                        obj_name = objs[cls]
                        if obj_name not in detections:
                            detections[obj_name] = []
                        r = box.xyxy
                        box_coords = [r[0][0], r[0][1], r[0][2], r[0][3]]
                        detections[obj_name].append(box_coords)

            return detections


# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 初始化视频捕获
    cap = cv2.VideoCapture(config.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.window_height)

    # 初始化检测器
    detector = MockDetector()
    #
    # # 初始化SOP监控器
    monitor = SOPMonitor(config)
    #
    print("SOP监控系统启动...")
    print("5道工序：")
    for i, step in enumerate(monitor.steps):
        print(f"  {i + 1}. {step.name} - {step.description}")
    print("\n按 'q' 退出，按 'r' 重置")

    # 主循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频流")
            break
        # # 执行检测（实际使用你的YOLO模型）
        detections = detector.detect(frame)
        #
        # # 更新工序状态q
        step_status = monitor.update(detections)
        #
        # # 绘制状态
        frame = monitor.draw_status(frame, detections)

        # 显示结果
        cv2.imshow('SOP Monitor', frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # 手动重置
            # monitor.current_step = 0
            # monitor.step_status = [False] * len(monitor.steps)
            # monitor.step_confirm_counter = [0] * len(monitor.steps)
            # monitor.step_start_time = None
            print("手动重置")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("SOP监控系统已关闭")


if __name__ == "__main__":
    main()
