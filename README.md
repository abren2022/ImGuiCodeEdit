from utile import AlertManager
import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from utile import RealtimeRecorder,AlertTypes
os.environ['ULTRALYTICS_VERBOSE'] = 'False'
recorder = RealtimeRecorder(buffer_seconds=10, fps=30)
alert_manager = AlertManager(max_history=50)
# 全局变量
camera = None




@dataclass
class Config():
    source_url: str =  ""
    model_path: str = r"./models/best.pt"  # 检测模型路径
    model_handkeys : str = r'./models/handkeys.pt' #手部关键点检测模型路径
    track_dist_threshold: float = 100.0
    rois = {
        'next_area': np.array([[2, 139], [64, 134], [68, 221], [4, 222]]),     # 下一道工序准备区域
        'pcb_area': np.array([[80, 76], [492, 80], [499, 296], [45, 296]]),  # 有效区域
        'label_area': np.array([[9, 186], [109, 191], [107, 324], [7, 334]]),  # 标签区域
        'buckle_area': np.array([[60, 21], [155, 18], [114, 141], [12, 135]]), # 扣件区域
    }

    # 目标类别ID
    class_ids = {
        'hand': 0,  # 人手
        'botake': 1,  # 壳体底座
        'halfpro': 2,  # pcb与底座壳体的半成品
        'product': 3,  # 盖板或者装配成品
        'label': 4,  # 标签
        'pcb': 5,  # PCB板子
        'scanlight': 6,  # 扫描灯光
        'buckle': 7
    }
    # 检测阈值
    conf_threshold = 0.6
    iou_threshold_contact: float = 0.01  # 判断接触的IOU阈值
    hold_frames: int = 2  # 动作确认需要的连续帧数
    show_fps: bool = True
    iou_match_thresh = 0.05
    track_lost_max_frames = 70

@dataclass
class ProcessStep:
    name: str
    description: str
    check_func: callable
    timeout: float = 30.0

@dataclass
class WorkpieceState:
    track_id: int
    step_status: List[bool] = field(default_factory=lambda: [False] * 8) # 对应8个步骤的状态
    current_step: int = 0          # 当前正在进行的步骤索引
    completed: bool = False        # 是否所有步骤已完成
    step_start_time: float = None  # 当前步骤开始时间
    step_confirm_counter: List[int] = field(default_factory=lambda: [0] * 8) # 用于防抖，确认步骤完成的连续帧数
    last_seen_frame: int = 0       # 最后被检测到的帧号

@dataclass
class ProductRecord:
    """单个产品的生产记录"""
    step_name= ["从料盘取出PCB", "正在将pcb板放入底座中", "PCB放入底座OK", "获取条码并扫描", "扫码PCB标签", "粘贴产品标签到顶盖", "组装盖板","安装压扣"]
    track_id: int
    start_time: float
    end_time: float = 0.0
    status: str = "InProgress"  # InProgress, Completed, Abnormal
    abnormal_reason: str = ""
    step_durations: Dict[int, float] = field(default_factory=dict) # 记录每个步骤的耗时


class ProductionStats:
    def __init__(self):
        self.total_produced = 0  # 总产量
        self.total_abnormal = 0  # 总异常数
        self.current_batch_records: List[ProductRecord] = []  # 当前批次记录
        self.start_time = time.time()  # 统计开始时间
        self.current_id = 0
        # 实时状态缓存 (用于UI显示最后完成的那个产品的状态，或者当前正在做的)
        self.last_completed_product: Optional[ProductRecord] = None
        self.current_active_products: Dict[int, ProductRecord] = {}  # 正在进行的工件

    def start_tracking(self, track_id: int, current_time: float):
        """开始跟踪一个新工件"""
        if track_id not in self.current_active_products:
            record = ProductRecord(
                track_id=track_id,
                start_time=current_time
            )
            self.current_active_products[track_id] = record

    def update_step_duration(self, track_id: int, step_index: int, duration: float):
        """更新某个步骤的耗时"""
        if track_id in self.current_active_products:
            self.current_id = track_id
            self.current_active_products[track_id].step_durations[step_index] = duration
            self.current_active_products[track_id].status = self.current_active_products[track_id].step_name[step_index]

    def complete_product(self, track_id: int, current_time: float, is_abnormal: bool = False, reason: str = ""):
        """完成一个工件的统计"""
        if track_id in self.current_active_products:
            record = self.current_active_products.pop(track_id)
            record.end_time = current_time
            record.status = "Abnormal" if is_abnormal else "Completed"
            record.abnormal_reason = reason

            self.total_produced += 1
            if is_abnormal:
                self.total_abnormal += 1

            self.last_completed_product = record
            self.current_batch_records.append(record)

            # 打印简要日志
            total_time = record.end_time - record.start_time
            status_str = "异常" if is_abnormal else "合格"
            print(f"[统计] 工件 {track_id} 完成 | 状态: {status_str} | 总耗时: {total_time:.2f}s")

    def get_abnormal_rate(self) -> float:
        """获取异常率"""
        if self.total_produced == 0:
            return 0.0
        return (self.total_abnormal / self.total_produced) * 100

    def get_current_status_summary(self) -> Dict[str, Any]:
        """获取当前整体统计摘要"""
        now = time.time()
        running_time = now - self.start_time

        # 计算平均单件耗时 (仅针对已完成的)
        avg_time = 0.0
        completed_records = [r for r in self.current_batch_records if r.status == "Completed"]
        if completed_records:
            total_duration = sum(r.end_time - r.start_time for r in completed_records)
            avg_time = total_duration / len(completed_records)

        return {
            "total_produced": self.total_produced,
            "abnormal_rate": f"{self.get_abnormal_rate():.2f}%",
            "avg_cycle_time": f"{avg_time:.2f}s",
            "running_time": f"{running_time / 60:.1f}min",
            "active_count": len(self.current_active_products)
        }
class SOPMonitor:
    def __init__(self, config: Config):
        self.config = config
        self.steps = self._init_steps()
        self.hand_detector = YOLO(Config.model_handkeys).to("cuda:3")
        self.frame_count = 0
        self.fps = 0
        self.text =""
        self.fps_update_time = time.time()
        self.workpieces: Dict[int, WorkpieceState] = {}
        self.stats_manager = ProductionStats()
        self.font = ImageFont.truetype('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc', 24)

    def _init_steps(self) -> List[ProcessStep]:
        """初始化6道工序"""
        return [
            ProcessStep("TakeWorkpiece", "从料盘取出PCB", self._check_take_workpiece),
            ProcessStep("PCBtakein", "将pcb板放入底座中", self._check_pcb_take_in),
            ProcessStep("PCBputedin", "PCB放入底座完成", self.is_assembly_completed),
            ProcessStep("ScanBarcode", "获取条码并扫码", self._check_scan),
            ProcessStep("ScanProduct", "扫码PCB标签", self._check_pcb_scan),
            ProcessStep("ApplyLabel", "粘贴产品标签", self._check_label),
            ProcessStep("Assemproduct", "组装盖板", self._check_assem_product),
            ProcessStep("Snaponbuckle", "拿取压扣", self._check_Snap_buckle),
        ]

    def get_or_create_workpiece(self, track_id: int) -> WorkpieceState:
        """获取或创建工件状态"""
        if track_id not in self.workpieces:
            self.workpieces[track_id] = WorkpieceState(track_id=track_id)
        return self.workpieces[track_id]

    def _find_object_by_id(self, tracked_objects: Dict[str, List[Any]], target_id: int, category: str) -> Optional[
        Dict]:
        """辅助函数：在指定类别列表中查找特定 ID 的对象"""
        if category not in tracked_objects:
            return None
        for obj in tracked_objects[category]:
            if obj.get('track_id') == target_id:
                return obj
        return None
    def update_all_workpieces(self, tracked_objects: Dict[str, List[Any]], current_time: float):
        """遍历所有检测到的工件并更新其状态"""
        # 1. 更新现有工件的状态
        active_ids = set()
        if current_time - self.fps_update_time > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_update_time = current_time
        workpiece_categories = ['pcb', 'product']
        for category in workpiece_categories:
            if category in tracked_objects:
                for wp_obj in tracked_objects[category]:
                    tid = wp_obj['track_id']
                    active_ids.add(tid)
                    wp_state = self.get_or_create_workpiece(tid)
                    wp_state.last_seen_frame = self.frame_count
                    self.stats_manager.start_tracking(tid, current_time)
                    # 调用原有的单工件更新逻辑
                    self._update_workpiece_step(tracked_objects, wp_state, current_time)

        # 2. 清理长时间未出现的工件
        ids_to_remove = []
        for tid, wp_state in self.workpieces.items():
            if tid not in active_ids and (self.frame_count - wp_state.last_seen_frame > 100):
                ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.workpieces[tid]

    def get_stats_overlay_dict(self):
        """生成用于显示的统计信息文本列表"""
        summary = self.stats_manager.get_current_status_summary()
        summary_ret = {"Total_Produced": summary['total_produced'], "Abnormal_Rate": summary['abnormal_rate'],"avg_cycle_time":summary['avg_cycle_time']}
        # 如果有正在进行的工件，显示其当前进度
        if self.stats_manager.current_active_products:
            # 取当前活跃工件
            cur_tid = self.stats_manager.current_id
            wp = self.workpieces.get(cur_tid)
            if wp:

                current_step_name = self.stats_manager.current_active_products[cur_tid].status
                progress = f"{wp.current_step}/{len(self.steps)}"
                summary_ret["progress"] = progress
                summary_ret["currentID"] = cur_tid
                summary_ret["current_step_name"] = current_step_name
        return summary_ret
    def _update_workpiece_step(self,tracked_objects: Dict[str, List[Any]], wp: WorkpieceState, current_time: float):
        """更新单个工件的工序状态"""
        # 1. 如果已完成所有工序，检查是否需要重置
        if all(wp.step_status):
            if not wp.completed:
                wp.completed = True
                self.text =u"工件 {} 完成所有工序".format(wp.track_id)
                self.stats_manager.complete_product(wp.track_id, current_time, is_abnormal=False)
            return
        # 检查当前工序
        if wp.current_step < len(self.steps):
            current_step_obj = self.steps[wp.current_step]

            # 设置开始时间
            if wp.step_start_time is None:
                wp.step_start_time = current_time

            # 检查超时
            if current_time - wp.step_start_time > current_step_obj.timeout:
                self.text =u"警告：工件 {} 工序 '{}' 超时".format(wp.track_id,current_step_obj.name)
                wp.step_start_time = current_time  # 重置计时器

            # 检查当前工序是否完成（需要关联到特定工件）
            if current_step_obj.check_func(tracked_objects, wp.track_id):
                wp.step_confirm_counter[wp.current_step] += 1
                if wp.step_confirm_counter[wp.current_step] >= self.config.hold_frames:
                    step_duration = current_time - wp.step_start_time
                    self.stats_manager.update_step_duration(wp.track_id, wp.current_step, step_duration)

                    wp.step_status[wp.current_step] = True
                    self.text = u"工件 {} 进度 {}/{} 正在操作: {}".format(wp.track_id,wp.current_step + 1,len(self.steps),current_step_obj.name)
                    print(self.text)
                    wp.current_step += 1
            else:
                wp.step_confirm_counter[wp.current_step] = max(0, wp.step_confirm_counter[wp.current_step] - 1)


    # ==================== 工序检查函数（需要关联到特定工件） ====================
    def _check_take_workpiece(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查是否完成取工件,主要判断在pcb区域是否存在pcb,且手和该pcb接触"""
        if 'hand' not in tracked_objects or 'pcb' not in tracked_objects:
            return False

        # 找到目标工件
        target_workpiece = self._find_object_by_id(tracked_objects, workpiece_id, 'pcb')
        if not target_workpiece:
            return False
        #workpiece_in_pcb_area = self._is_in_roi(target_workpiece['bbox'], 'pcb_area')
        # 检查手是否接触该工件
        hand_holding_pcb = False
        for hand in tracked_objects['hand']:
            iou = self._calculate_iou(hand['bbox'], target_workpiece['bbox'])
            if iou > self.config.iou_threshold_contact:
                hand_holding_pcb = True

        return hand_holding_pcb

    def _check_pcb_take_in(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查PCB装入底板的操作，手拿PCB靠近底板"""
        required = ['hand', 'botake', 'pcb']
        if not all(k in tracked_objects for k in required):
            return False

        # 找到目标工件
        target_workpiece = self._find_object_by_id(tracked_objects, workpiece_id, 'pcb')

        if not target_workpiece:
            return False

        hand_holding_pcb = False
        pcb_near_botake = False
        try:
            for hand in tracked_objects['hand']:
                if self._calculate_iou(hand['bbox'], target_workpiece['bbox']) > self.config.iou_threshold_contact:
                    hand_holding_pcb = True
                    break

            for botake in tracked_objects['botake']:
                dist = self._calculate_iou(botake['bbox'], target_workpiece['bbox'])
                if dist > self.config.iou_threshold_contact:
                    pcb_near_botake = True
                    break

            return hand_holding_pcb or pcb_near_botake
        except:
            print("waring:'NoneType' object is not subscriptable")
            return False

    def is_assembly_completed(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查是否完成装配"""
        required = ['halfpro']
        if not all(k in tracked_objects for k in required):
            return False
        return True

    def _check_scan(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查手进入标签区域获取标签，同时应该存在半成品，否则属于其他操作"""
        required = ['hand','halfpro']
        if not all(k in tracked_objects for k in required):
            return False
        workpiece_in_label_area =  False
        for hand in tracked_objects['hand']:
            workpiece_in_label_area = self._is_in_roi(hand['bbox'], 'label_area')
            if workpiece_in_label_area:
                break
        return workpiece_in_label_area

    def _check_pcb_scan(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查是否进行pcb 扫码操作，只要手持PCB进入扫码区域"""
        required = ['hand', 'pcb']
        if not all(k in tracked_objects for k in required):
            return False

        hand_holding_pcb = False
        target_workpiece = self._find_object_by_id(tracked_objects, workpiece_id, 'pcb')
        for hand in tracked_objects['hand']:
            if self._calculate_iou(hand['bbox'], target_workpiece['bbox']) > self.config.iou_threshold_contact:
                hand_holding_pcb = True
                break
        workpiece_in_label_area = self._is_in_roi(target_workpiece['bbox'], 'label_area')
        return hand_holding_pcb and workpiece_in_label_area

    def _check_label(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查是否完成贴标签，双手持PCB"""
        required = ['hand', 'halfpro', 'product', 'pcb']
        if not all(k in tracked_objects for k in required):
            return False

        # 找到目标工件
        target_workpiece = None
        for wp in tracked_objects['product']:
            if wp['track_id'] == workpiece_id:
                target_workpiece = wp
                break

        if not target_workpiece:
            return False
        hands_holding_count = 0
        # 判断：标签接触目标工件
        for hand in tracked_objects['hand']:
            iou = self._calculate_iou(hand['bbox'], target_workpiece['bbox'])
            if iou > self.config.iou_threshold_contact:
                hands_holding_count += 1

        return hands_holding_count >= 2

    def _check_assem_product(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int) -> bool:
        """检查是否完成盖板安装工序"""
        required = ['hand', 'product', 'label']
        if not all(k in tracked_objects for k in required):
            return False

        # 找到目标工件
        target_workpiece = None
        for wp in tracked_objects['product']:
            if wp['track_id'] == workpiece_id:
                target_workpiece = wp
                break

        if not target_workpiece:
            return False

        hands_holding_count = 0
        # 判断：标签接触目标工件
        for hand in tracked_objects['hand']:
            iou = self._calculate_iou(hand['bbox'], target_workpiece['bbox'])
            if iou > self.config.iou_threshold_contact:
                hands_holding_count += 1

        return hands_holding_count >= 2

    def _check_Snap_buckle(self,tracked_objects: Dict[str, List[Any]], workpiece_id: int)-> bool:
        """检查是否完成 Snap Buckle"""
        required = ['hand', 'product', 'label']
        if not all(k in tracked_objects for k in required):
            return False
        target_workpiece = None
        for wp in tracked_objects['product']:
            if wp['track_id'] == workpiece_id:
                target_workpiece = wp
                break

        if not target_workpiece:
            return False
        take_buckle = False
        hand_take_product = False
        for hand in tracked_objects['hand']:
            workpiece_in_label_area = self._is_in_roi(hand['bbox'], 'buckle_area')
            if workpiece_in_label_area:
                take_buckle = True

        for hand in tracked_objects['hand']:
            iou = self._calculate_iou(hand['bbox'], target_workpiece['bbox'])
            if iou > self.config.iou_threshold_contact:
                hand_take_product =  True
        return  take_buckle and hand_take_product

    def _assem_Snap_buckle(self, tracked_objects: Dict[str, List[Any]], workpiece_id: int):
        """检查是否安装 Snap Buckle"""
        required = ['buckle', 'product']
        if not all(k in tracked_objects for k in required):
            return False
        return True
    # ==================== 辅助函数 ====================
    def _calculate_iou(self, box1, box2):
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

    def normalize_coordinates_if_needed(self, keypoints, img_width, img_height):
        """
        自动检测坐标格式并归一化

        支持两种格式:
        1. 归一化坐标 (0-1) -> 直接使用
        2. 绝对像素坐标 -> 转换为归一化坐标

        Returns:
            normalized_keypoints: 归一化后的关键点 (21, 3)
        """
        # 检查坐标范围
        xy_coords = keypoints[:, :2]
        max_coord = np.max(xy_coords)

        if max_coord > 1.0:
            # 坐标是绝对像素值，需要归一化
            normalized = keypoints.copy()
            normalized[:, 0] = keypoints[:, 0] / img_width
            normalized[:, 1] = keypoints[:, 1] / img_height
            return normalized
        else:
            # 已经是归一化坐标
            print(f"检测到归一化坐标 (最大值={max_coord:.2f})")
            return keypoints

    def draw_hand_skeleton(self, image, keypoints, visibility_threshold=0.6):
        """
        在图像上绘制手部骨架

        Args:
            image: 输入图像（numpy数组）
            keypoints: 关键点坐标和可见性，shape为(21, 3)，每个点包含[x, y, visibility]
            visibility_threshold: 可见性阈值，大于此值才绘制
        """
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        keypoints_norm = self.normalize_coordinates_if_needed(keypoints, w, h)
        # 将归一化坐标转换为像素坐标
        points_pixel = []
        visible_flags = []
        HAND_CONNECTIONS = [
            # 手腕到手掌
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            # 拇指 (1-4)
            (1, 2), (2, 3), (3, 4),
            # 食指 (5-8)
            (5, 6), (6, 7), (7, 8),
            # 中指 (9-12)
            (9, 10), (10, 11), (11, 12),
            # 无名指 (13-16)
            (13, 14), (14, 15), (15, 16),
            # 小指 (17-20)
            (17, 18), (18, 19), (19, 20)
        ]
        for i, kpt in enumerate(keypoints_norm):
            x_norm, y_norm, visibility = kpt
            # 检查有效性：坐标在[0,1]范围内，且可见性大于阈值
            is_valid = (0 <= x_norm <= 1) and (0 <= y_norm <= 1) and visibility >= visibility_threshold

            if is_valid:
                px = int(x_norm * w)
                py = int(y_norm * h)
                points_pixel.append((px, py))
                visible_flags.append(True)
            else:
                points_pixel.append(None)
                visible_flags.append(False)

        # 绘制关键点连接线（骨架）
        # for connection in HAND_CONNECTIONS:
        #     start_idx, end_idx = connection
        #     if start_idx < len(points_pixel) and end_idx < len(points_pixel):
        #         start_point = points_pixel[start_idx]
        #         end_point = points_pixel[end_idx]
        #
        #         if start_point is not None and end_point is not None:
        #             cv2.line(img_copy, start_point, end_point, (0, 255, 0), 1)  # 绿色线条

        # 绘制关键点
        for i, (point, is_visible) in enumerate(zip(points_pixel, visible_flags)):
            if point is not None:
                # 根据可见性使用不同颜色：可见=红色，遮挡=橙色
                color = (0, 0, 255) if is_visible else (0, 165, 255)  # BGR格式
                cv2.circle(img_copy, point, 1, color, -1)

        return img_copy

    def center_bt_distance(self, box1, box2):
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _is_in_roi(self, box, roi_name: str) -> bool:
        if roi_name not in self.config.rois:
            return False

        roi = self.config.rois[roi_name]
        box_center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

        return cv2.pointPolygonTest(roi, box_center, False) >= 0

    # ==================== 绘制函数 ====================
    def draw_status(self, frame: np.ndarray, tracked_objects: Dict[str, List[Any]]) -> np.ndarray:
        """在图像上绘制所有工件的状态"""
        results = self.hand_detector.predict(frame, verbose=False)
        result = results[0]
        keypoints_data = result.keypoints.data.cpu().numpy()
        for hand_idx, keypoints in enumerate(keypoints_data):
            frame = self.draw_hand_skeleton(frame, keypoints)
        # 绘制检测框和跟踪ID
        colors = {
            'hand': (0, 255, 0),
            'botake': (255, 0, 0),
            'halfpro': (0, 0, 255),
            'product': (255, 255, 0),
            'label': (255, 0, 255),
            'pcb': (0, 255, 255),
            'scanlight': (255, 255, 255),
            'buckle': (255, 0, 255),
        }
        for obj_name in ['label', 'botake', 'halfpro', 'scanlight', 'buckle']:
            if obj_name in tracked_objects:
                for obj in tracked_objects[obj_name]:
                    bbox = obj['bbox']
                    x1, y1, x2, y2 = map(int, (bbox[0], bbox[1], bbox[2], bbox[3]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[obj_name], 2)
                    cv2.putText(frame, obj_name, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[obj_name], 1)
        if 'pcb' in tracked_objects:
            for wp in tracked_objects['pcb']:
                bbox = wp['bbox']
                track_id = wp['track_id']
                x1, y1, x2, y2 = map(int, (bbox[0], bbox[1], bbox[2], bbox[3]))
                color = colors['pcb']
                thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 显示跟踪ID
                label = f"{track_id}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if 'product' in tracked_objects:
            for wp in tracked_objects['product']:
                bbox = wp['bbox']
                track_id = wp['track_id']
                x1, y1, x2, y2 = map(int, (bbox[0], bbox[1], bbox[2], bbox[3]))
                color = colors['product']
                thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 显示跟踪ID
                label = f"{track_id}"
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)
        draw.text((66, 320), self.text, font=self.font,fill=(0, 255, 0))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame

# ==================== YOLO检测器（带跟踪） ====================
class YOLODetector:
    """使用Ultralytics YOLO进行目标检测，只对工件进行跟踪"""

    def __init__(self, config: Config):
        self.config = config
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(config.model_path).to(device)
        self.model.eval()
        # 所有需要检测的类别ID（工件+其他）
        self.all_detect_class_ids = list(config.class_ids.values())
        # 类别ID到名称的反向映射（避免循环匹配）
        self.id2name = {cid: name for name, cid in config.class_ids.items()}
        self.current_frame = 0  # 帧计数器
        self.next_business_id = 1
        self.active_objects: Dict[int, Dict[str, Any]] = {}
        self.pcb_to_product_map = {} #跟踪与状态继承

    @staticmethod
    def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """计算两个归一化bbox的IOU（交并比）"""
        h, w = 360, 640
        x1_1, y1_1, x2_1, y2_1 = [int(v * w) for v in bbox1[:2]] + [int(v * h) for v in bbox1[2:]]
        x1_2, y1_2, x2_2, y2_2 = [int(v * w) for v in bbox2[:2]] + [int(v * h) for v in bbox2[2:]]

        # 计算交集
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _is_in_region(self, box, roi_name: str) -> bool:
        if roi_name not in self.config.rois:
            return False

        roi = self.config.rois[roi_name]
        box_center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

        return cv2.pointPolygonTest(roi, box_center, False) >= 0

    def center_distance(self, box1, box2):
        c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
        c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def _get_center(self, bbox: List[float]) -> list[float]:
        """获取 bbox 中心点"""
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def box_is_in_roi(self, box, roi_name: str) -> bool:
        if roi_name not in self.config.rois:
            return False

        roi = self.config.rois[roi_name]
        box_center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

        return cv2.pointPolygonTest(roi, box_center, False) >= 0


    def _match_active_object(self, bbox: List[float], target_type: str) -> Optional[int]:
        """在活跃对象中查找同类型的最近对象,就是来一个新的，需要从历史中寻找距离相近的同类进行配对"""
        min_dist = float('inf')
        matched_id = None
        curr_cx, curr_cy = self._get_center(bbox)

        for bid, info in self.active_objects.items():
            if info['type'] != target_type:
                continue
            last_cx, last_cy = self._get_center(info['last_bbox'])
            dist = np.sqrt((curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2)
            if dist < self.config.track_dist_threshold*3 and dist < min_dist:
                min_dist = dist
                matched_id = bid
        return matched_id


    def judge_product_regine(self, target_type: str):
        workpiece_in_area = False
        for bid, info in self.active_objects.items():
            if info['type'] != target_type:
                continue
            box_info = info['last_bbox']
            workpiece_in_area = self._is_in_region(box_info, 'pcb_area')
        return workpiece_in_area

    def _create_new_object(self, bbox: List[float], obj_type: str) -> int:
        new_id = self.next_business_id
        self.next_business_id += 1
        self.active_objects[new_id] = {
            'last_bbox': bbox,
            'last_seen_frame': self.current_frame,
            'type': obj_type
        }
        return new_id

    def _update_object_state(self, obj_id: int, bbox: List[float], new_type: str = None):
        if obj_id in self.active_objects:
            self.active_objects[obj_id]['last_bbox'] = bbox
            self.active_objects[obj_id]['last_seen_frame'] = self.current_frame
            if new_type:
                self.active_objects[obj_id]['type'] = new_type

    def _cleanup_inactive_objects(self):
        ids_to_remove = []
        for bid, info in self.active_objects.items():
            if self.current_frame - info['last_seen_frame'] > self.config.track_lost_max_frames:
                ids_to_remove.append(bid)
        for bid in ids_to_remove:
            del self.active_objects[bid]

    def detect_and_track(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        global recorder
        if recorder:
            recorder.update(frame)
        results_dict = defaultdict(list)
        self.current_frame += 1

        results = self.model.predict(frame, verbose=False)

        if results[0].boxes is None:
            return dict(results_dict)

        boxes = results[0].boxes.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)


        for idx, (box, class_id, conf) in enumerate(zip(boxes, class_ids, confs)):
            if confs[idx] < self.config.conf_threshold:
                continue
            obj_name = self.id2name.get(class_id, None)
            if not obj_name:
                continue

            r = box.xyxy
            bbox = [float(r[0][0]), float(r[0][1]), float(r[0][2]), float(r[0][3])]

            # 基础结果
            result_item = {
                'bbox': bbox,
                'conf': float(conf),
                'class_name': obj_name,
                'track_id': -1
            }
            # --- 核心跟踪逻辑 ---
            assigned_id = None

            if obj_name == 'pcb':
                # 1. PCB 匹配逻辑：找最近的活跃 PCB
                matched_id = self._match_active_object(bbox, 'pcb')
                if matched_id:
                    assigned_id = matched_id
                    self._update_object_state(assigned_id, bbox)
                else:
                    min_dist = float('inf')
                    for pid, info in self.active_objects.items():
                        if info['type'] == 'product':
                            dist = self.center_distance(bbox, info['last_bbox'])
                            if dist < self.config.track_dist_threshold*2 and dist < min_dist:
                                min_dist = dist
                                assigned_id = pid
                    if not assigned_id:
                        assigned_id = self._create_new_object(bbox, 'pcb')

            elif obj_name == 'product':
                # 2. Product 匹配逻辑：
                # A. 先找是否有活跃的 product (防止多个产品同时存在)
                matched_prod_id = self._match_active_object(bbox, 'product')
                if matched_prod_id:
                    assigned_id = matched_prod_id
                    self._update_object_state(assigned_id, bbox)
                else:
                    # B. 如果没有活跃 product，检查是否由刚消失的 PCB 转变而来
                    # 查找最近消失的 PCB (last_seen_frame 接近当前帧，且距离近)
                    inherited_id = self._match_active_object(bbox, 'pcb')
                    if inherited_id:
                        # 继承 PCB 的 ID！这是关键
                        assigned_id = inherited_id
                        # 将该 ID 的类型标记更新为 product，或者保留原状但更新位置
                        # 这里我们将 active_objects 中的类型更新，或者直接复用 ID
                        self._update_object_state(assigned_id, bbox, new_type='product')
                        print(f"✅ ID {assigned_id} 从 PCB 转变为 Product")
                    else:
                        continue

            # 将 ID 写入结果
            if assigned_id is not None:
                result_item['track_id'] = assigned_id
            results_dict[obj_name].append(result_item)


        self._cleanup_inactive_objects()
        return dict(results_dict)

