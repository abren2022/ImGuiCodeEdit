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
        self.active_objects_product: Dict[int, Dict[str, Any]] = {}#跟踪与状态继承

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
        workpiece_in_area = self._is_in_region(bbox, 'pcb_area')
        for bid, info in self.active_objects.items():
            if info['type'] != target_type:
                continue
            activate_in_area = self._is_in_region(info['last_bbox'], 'pcb_area')
            if workpiece_in_area and activate_in_area:
                matched_id = bid
                break
            elif workpiece_in_area and not activate_in_area:
                break
            else:
                last_cx, last_cy = self._get_center(info['last_bbox'])
                dist = np.sqrt((curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2)
                if dist < self.config.track_dist_threshold*3 and dist < min_dist:
                    min_dist = dist
                    matched_id = bid
        return matched_id

    def _match_active_product(self, bbox: List[float]) -> Optional[int]:
        """在活跃对象中查找同类型的最近对象,就是来一个新的，需要从历史中寻找距离相近的同类进行配对"""
        min_dist = float('inf')
        matched_id = None
        curr_cx, curr_cy = self._get_center(bbox)

        for bid, info in self.active_objects_product.items():
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

    def _copy_id_new_object(self,bbox: List[float], reid:int):
        if reid in self.active_objects_product:
            self.active_objects_product[reid]['last_bbox'] = bbox
            self.active_objects_product[reid]['last_seen_frame'] = self.current_frame
        else:
            self.active_objects_product[reid] = {
                'last_bbox': bbox,
                'last_seen_frame': self.current_frame,
                'type': 'product'
            }
        return reid
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

    def _cleanup_inactive_product(self):
        ids_to_remove = []
        for bid, info in self.active_objects_product.items():
            if self.current_frame - info['last_seen_frame'] > self.config.track_lost_max_frames:
                ids_to_remove.append(bid)
        for bid in ids_to_remove:
            del self.active_objects_product[bid]

    def detect_and_track(self, frame: np.ndarray) -> Dict[str, List[Dict]]:
        global recorder
        self.current_frame += 1
        if recorder and self.current_frame%2 == 0:
            recorder.update(frame)
        results_dict = defaultdict(list)
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
                    for pid, info in self.active_objects_product.items():
                        dist = self.center_distance(bbox, info['last_bbox'])
                        if dist < self.config.track_dist_threshold*2 and dist < min_dist:
                            min_dist = dist
                            assigned_id = pid
                    if not assigned_id:
                        assigned_id = self._create_new_object(bbox, 'pcb')

            # elif obj_name == 'product':
            #     # 2. Product 匹配逻辑：
            #     # A. 先找是否有活跃的 product (防止多个产品同时存在)
            #     matched_prod_id = self._match_active_object(bbox, 'product')
            #     if matched_prod_id:
            #         assigned_id = matched_prod_id
            #         self._update_object_state(assigned_id, bbox)
            #     else:
            #         # B. 如果没有活跃 product，检查是否由刚消失的 PCB 转变而来
            #         # 查找最近消失的 PCB (last_seen_frame 接近当前帧，且距离近)
            #         inherited_id = self._match_active_object(bbox, 'pcb')
            #         if inherited_id:
            #             # 继承 PCB 的 ID！这是关键
            #             assigned_id = inherited_id
            #             # 将该 ID 的类型标记更新为 product，或者保留原状但更新位置
            #             # 这里我们将 active_objects 中的类型更新，或者直接复用 ID
            #             self._update_object_state(assigned_id, bbox, new_type='product')
            #             print(f"✅ ID {assigned_id} 从 PCB 转变为 Product")
            #         else:
            #             continue

            elif obj_name == 'product':
                # 1. 优先尝试从 PCB 继承 ID（新出现的产品）
                inherited_id = self._match_active_object(bbox, 'pcb')
                if inherited_id is not None:
                    assigned_id = inherited_id
                    # 继承后更新类型为 product，并更新位置
                    self._copy_id_new_object(bbox, assigned_id)
                else:
                    # 2. 继承失败，再匹配已有的 product（已存在的产品继续跟踪）
                    matched_prod_id = self._match_active_product(bbox)
                    if matched_prod_id is not None:
                        assigned_id = matched_prod_id
                        self._copy_id_new_object(bbox,assigned_id)
                    else:
                        continue
            # 将 ID 写入结果
            if assigned_id is not None:
                result_item['track_id'] = assigned_id
            results_dict[obj_name].append(result_item)


        self._cleanup_inactive_objects()
        self._cleanup_inactive_product()
        return dict(results_dict)
