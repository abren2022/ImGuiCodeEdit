import cv2
from dataclasses import dataclass, field
from collections import deque
import time
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import os
import queue
import threading

class VideoSaver:
    """视频保存器，支持多种编码格式和压缩选项"""

    def __init__(self, fps=25, codec='h264'):
        """
        Args:
            fps: 帧率
            codec: 编码器 ('h264', 'mp4v', 'avc1')
        """
        self.fps = fps
        self.codec_map = {
            'avc1': cv2.VideoWriter.fourcc(*'avc1'),
        }
        self.fourcc = self.codec_map.get(codec, cv2.VideoWriter.fourcc(*'avc1'))

    def save_frames(self, frames, output_path, is_color=True):
        """保存帧列表为视频"""
        if not frames:
            return False

        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, self.fourcc, self.fps, (w, h), is_color)

        for frame in frames:
            out.write(frame)

        out.release()
        return True



class RealtimeRecorder:
    def __init__(self, buffer_seconds=10, fps=30):
        self.saver = VideoSaver(fps=fps, codec='mp4v')
        self.frame_buffer = deque(maxlen=fps * buffer_seconds)  # 环形缓冲区，只保留最近N秒
        self.timestamps =  time.time()
        self.is_recording_trigger = False
        self.save_queue = queue.Queue(maxsize=5)  # 队列限制大小，防止内存爆炸
        self.is_running = True
        # 启动后台保存线程
        self.save_thread = threading.Thread(target=self._background_saver, daemon=True)
        self.save_thread.start()

    def update(self, frame):
        """每帧调用"""
        current_time = time.time()
        self.frame_buffer.append(frame.copy())  # 必须copy，否则后续帧会覆盖
        self.timestamps=current_time

    def save_alarm_clip(self, filename_prefix="alarm", save_dir="./alert_videos"):
        """当触发报警时，保存缓冲区内的视频"""
        if len(self.frame_buffer) > 0:
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                    print(f"创建报警视频目录: {save_dir}")
                except Exception as e:
                    print(f"创建目录失败: {e}")
                    return None
            # 将deque转换为list
            frames_list = list(self.frame_buffer)
            now_str = datetime.fromtimestamp(self.timestamps).strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{now_str}.mp4"
            full_path = os.path.join(save_dir, filename)
            task_data = {
                'frames': frames_list,
                'timestamp': now_str,
                'prefix': filename_prefix,
                'dir': save_dir
            }
            try:
                self.save_queue.put_nowait(task_data)
                # print("📹 报警视频保存任务已提交至后台队列")
                return full_path  # 返回 pending 表示正在后台处理，具体路径由后台生成
            except queue.Full:
                print("⚠️ 警告：视频保存队列已满，跳过此次报警视频保存以避免内存溢出")
                return None

    def _background_saver(self):
        """后台线程：从队列获取任务并保存视频"""
        while self.is_running:
            try:
                # 阻塞等待任务，timeout防止线程无法退出
                task = self.save_queue.get(timeout=1)

                frames_list = task['frames']
                ts = task['timestamp']
                prefix = task['prefix']
                save_dir = task['dir']

                # 执行实际的保存操作 (耗时操作在后台进行)
                self._do_save(frames_list, ts, prefix, save_dir)

                self.save_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"后台视频保存线程出错: {e}")

    def _do_save(self, frames_list, timestamp, prefix, save_dir):
        """实际执行保存的逻辑"""
        if not frames_list:
            return

        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                print(f"创建目录失败: {e}")
                return

        now_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{now_str}.mp4"
        full_path = os.path.join(save_dir, filename)

        try:
            # 这里调用原有的 saver 逻辑
            ret = self.saver.save_frames(frames_list,full_path)
            if ret:
                print(f"✅ 后台保存完成: {filename}")
            else:
                print(f"❌ 后台保存失败: {filename}")
        except Exception as e:
            print(f"保存视频文件时发生异常: {e}")

    def stop(self):
        """停止后台线程"""
        self.is_running = False
        if self.save_thread.is_alive():
            self.save_thread.join()

class AlertTypes:
    PLACEMENT_ERROR = "placement_error"   # 放置错误
    TIMEOUT_ERROR = "timeout_error"       # 超时错误
    MISSING_STEP = "missing_step"         # 工序缺失
    SEQUENCE_ERROR = "sequence_error"     # 顺序错误
    WRONG_STEP = "wrong_step"

@dataclass
class AlertEvent:
    """单个异常事件记录"""
    id: int           # 唯一ID
    timestamp: float  # 发生时间戳 (time.time())
    alert_type: str   # 异常类型
    description: str  # 详细描述
    video_clip_path: Optional[str] = None  # 对应的视频片段路径或内存引用
    duration: float = 0.0  # 异常持续时长（如果是持续性异常）

    @property
    def datetime_str(self) -> str:
        """获取可读的时间字符串"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


class AlertManager:
    """异常数据管理器"""

    def __init__(self, max_history: int = 100):
        self.alerts: List[AlertEvent] = []
        self.max_history = max_history
        self.total_count = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        # 实时统计缓存
        self._current_alert_type_counts = {}


    def _remove_video_file(self, video_path: Optional[str]):
        """安全地删除视频文件"""
        if not video_path:
            return

        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"已清理过期报警视频: {video_path}")
            else:
                print(f"尝试删除不存在的视频文件: {video_path}")
        except Exception as e:
            print(f"删除视频文件失败 {video_path}: {e}")

    def add_alert(self, alert_type: str, description: str,
                  video_clip_path: Optional[str] = None,) -> AlertEvent:
        """添加一个新的异常记录"""
        self.total_count += 1
        event_id = self.total_count

        new_alert = AlertEvent(
            id=event_id,
            timestamp=time.time(),
            alert_type=alert_type,
            description=description,
            video_clip_path=video_clip_path,
        )
        with self.lock:
            self.alerts.append(new_alert)

        # 更新类型计数
        self._current_alert_type_counts[alert_type] = self._current_alert_type_counts.get(alert_type, 0) + 1

        # 保持历史记录长度限制
        if len(self.alerts) > self.max_history:
            oldest_alert = self.alerts[0]
            removed_video_path = oldest_alert.video_clip_path
            self.alerts.pop(0)
            self._remove_video_file(removed_video_path)
            # 如果移除的是最旧的，可能需要重新计算计数，或者简单起见只保留最近N条的统计
            # 这里简化处理：只统计最近 max_history 条
            self._recalculate_stats()
        return new_alert

    def _recalculate_stats(self):
        """重新计算最近 max_history 条记录的统计信息"""
        self._current_alert_type_counts = {}
        for alert in self.alerts[-self.max_history:]:
            self._current_alert_type_counts[alert.alert_type] = \
                self._current_alert_type_counts.get(alert.alert_type, 0) + 1

    def get_statistics(self) -> dict:
        """获取当前统计摘要"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if current_time > self.start_time else 1

        # 计算异常率 (次/分钟 或 次/小时，根据需求调整)
        rate_per_minute = (len(self.alerts) / elapsed_time) * 60 if elapsed_time > 0 else 0

        return {
            'total_count': self.total_count,
            'recent_count': len(self.alerts),
            'rate_per_minute': round(rate_per_minute, 2),
            'type_distribution': self._current_alert_type_counts.copy(),
            'last_alert_time': self.alerts[-1].datetime_str if self.alerts else None
        }

    def get_recent_alerts(self, limit: int = 10) -> List[AlertEvent]:
        """获取最近的异常列表"""
        with self.lock:
            return self.alerts[-limit:]




. : 无法加载文件 C:\Windows\System32\WindowsPowerShell\v1.0\profile.ps1，因为在此系统上禁止运行脚本。有关详细信息，请参
阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 about_Execution_Policies。
所在位置 行:1 字符: 3
+ . 'C:\Windows\System32\WindowsPowerShell\v1.0\profile.ps1'
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) []，PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
. : 无法加载文件 C:\Users\yao.jie10\Documents\WindowsPowerShell\profile.ps1，因为在此系统上禁止运行脚本。有关详细信息，
请参阅 https:/go.microsoft.com/fwlink/?LinkID=135170 中的 about_Execution_Policies。
所在位置 行:1 字符: 3
+ . 'C:\Users\yao.jie10\Documents\WindowsPowerShell\profile.ps1'
+   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : SecurityError: (:) []，PSSecurityException
    + FullyQualifiedErrorId : UnauthorizedAccess
from flask import Flask, render_template, Response, jsonify, url_for, request
from flask import send_from_directory, abort,send_file
import random
import os
import json
from datetime import datetime, timedelta
import cv2
import urllib.parse
import time
import mimetypes
import threading
from AsseBehaviorMonitoring import Config
import gc
app = Flask(__name__)
alert_data = {
    'count': 522,
    'percentage': 1.1,
    'in_zone_time': 0.8,
    'motion_detected': False,
    'targets': {'person': 5, 'vehicle': 3, 'bicycle': 1}
}

CAMERAS_CONFIG_FILE = 'cameras.json'
LOCAL_VIDEO_DIR = 'D:/project/actiondetection/local_videos'
# 最终处理完成的帧缓存
LAST_PROCESSED_FRAME = None
# 线程锁：保护帧、摄像头读写
FRAME_LOCK = threading.Lock()
# 后台线程启停标记
BACKEND_RUNNING = True
# 全局摄像头管理器
class CameraManager:
    def __init__(self, config_file):
        self.config_file = config_file
        self.cameras = []          # 摄像头配置列表
        self.current_camera = None # 唯一的 VideoCamera 实例
        self.active_camera_id = None
        self.load_from_file()
        # 如果有摄像头，初始化当前实例并激活第一个
        if self.cameras:
            self._init_current_camera(self.cameras[0]['id'])

    def _init_current_camera(self, camera_id):
        """根据摄像头id初始化唯一的摄像头实例"""
        cam_info = next((c for c in self.cameras if c['id'] == camera_id), None)
        if not cam_info:
            return False
        if self.current_camera:
            self.current_camera.del_cap()
        self.current_camera = VideoCamera(cam_info['rtsp'])
        self.active_camera_id = int(camera_id)
        return True

    def load_from_file(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.cameras = json.load(f)
            print(f"加载了 {len(self.cameras)} 个摄像头")
        else:
            print("配置文件不存在，创建默认摄像头")
            self.cameras = []
            self._save_to_file()

    def _save_to_file(self):
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.cameras, f, indent=2, ensure_ascii=False)

    def get_list(self):
        return [{'id': c['id'], 'name': c['name'], 'status': c.get('status', 'online')}
                for c in self.cameras]

    def add_camera(self, name, rtsp_url):
        new_id = max([c['id'] for c in self.cameras], default=0) + 1
        new_cam = {'id': new_id, 'name': name, 'rtsp': rtsp_url, 'status': 'online'}
        self.cameras.append(new_cam)
        self._save_to_file()
        return new_id

    def switch_to(self, camera_id):
        camera_id = int(camera_id)
        if self.active_camera_id == camera_id:
            print("已是当前摄像头")
            return True, "已是当前摄像头"

        cam_info = next((c for c in self.cameras if c['id'] == camera_id), None)
        if not cam_info:
            return False, "摄像头不存在"
        # 复用同一个 VideoCamera 实例，只重新配置 RTSP
        if not self.current_camera:
            self.current_camera = VideoCamera(cam_info['rtsp'])
        else:
            success, msg = self.current_camera.reconfigure(cam_info['rtsp'])
            if not success:
                return False, msg
        self.active_camera_id = camera_id
        return True, "切换成功"

    def reconfigure_active(self, new_rtsp_url, new_name=None):
        if self.active_camera_id is None:
            return False, "没有激活的摄像头"
        for cam in self.cameras:
            if cam['id'] == self.active_camera_id:
                cam['rtsp'] = new_rtsp_url
                if new_name:
                    cam['name'] = new_name
                self._save_to_file()
                if self.current_camera:
                    success, msg = self.current_camera.reconfigure(new_rtsp_url)
                    return success, msg
                else:
                    self.current_camera = VideoCamera(new_rtsp_url)
                    return True, "重新配置成功"
        return False, "未找到摄像头"

    def get_active_frame(self):
        if self.current_camera:
            return self.current_camera.get_frame()
        return None


class VideoCamera:
    def __init__(self, rtsp_url):
        self.video_source = rtsp_url
        self.cap = None
        self.lock = threading.RLock()
        self.is_simulated = False
        self.initialize_camera()

    def initialize_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        try:
            print(f"正在连接 RTSP: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise Exception("无法打开")
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise Exception("无法读取帧")
            self.is_simulated = False
            print("摄像头连接成功")
        except Exception as e:
            print(f"摄像头初始化失败: {e}，使用模拟模式")
            self.is_simulated = True
            self.cap = None

    def reconfigure(self, new_rtsp_url):
        """复用实例，切换 RTSP 流"""
        # 释放旧连接
        #
        if self.cap:
            self.cap.release()
            self.cap = None
        time.sleep(0.3)
        gc.collect()
        self.video_source = new_rtsp_url
        try:
            self.cap = cv2.VideoCapture(new_rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                self.is_simulated = True
                return False, "无法打开RTSP流"
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_simulated = True
                return False, "无法读取视频帧"
            self.is_simulated = False
            print(f"成功切换到新摄像头: {new_rtsp_url}")
            return True, "切换成功"
        except Exception as e:
            self.is_simulated = True
            return False, str(e)

    def get_frame(self):
        #with self.lock:
        if not self.is_simulated and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        return None

    def del_cap(self):
        #with self.lock:
        if self.cap:
            self.cap.release()
            self.cap = None
            time.sleep(2)

def scan_local_videos():
    """扫描本地视频目录，生成事件列表"""
    events = []
    if not os.path.exists(LOCAL_VIDEO_DIR):
        os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)
        return events

    supported_ext = ('.mp4', '.avi', '.mov', '.mkv')
    for filename in os.listdir(LOCAL_VIDEO_DIR):
        if filename.lower().endswith(supported_ext):
            filepath = os.path.join(LOCAL_VIDEO_DIR, filename)
            mtime = os.path.getmtime(filepath)
            event_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
            # 根据文件名简单推断事件类型
            event_type = "motion"
            if "person" in filename.lower():
                event_type = "person"
            elif "vehicle" in filename.lower():
                event_type = "vehicle"
            elif "bicycle" in filename.lower():
                event_type = "bicycle"

            events.append({
                'type': event_type,
                'time': event_time,
                'camera': '本地存档',
                'description': filename,
                'confidence': 85,
                'video_url': f'/video_clips/{filename}'   # 可访问的URL
            })
    return events
def backend_stream_loop():
    global LAST_PROCESSED_FRAME
    while BACKEND_RUNNING:
        try:
            frame = camera_manager.get_active_frame()
            if frame is None:
                time.sleep(0.03)
                continue

            # 全局唯一一次AI检测 + 绘制
            # detections = detector.detect_and_track(frame)
            # current_time = time.time()
            # monitor.update_all_workpieces(detections, current_time)
            # frame = monitor.draw_status(frame, detections)

            # 更新全局帧缓存，加锁保护
            with FRAME_LOCK:
                LAST_PROCESSED_FRAME = frame.copy()

        except Exception as e:
            print(f"后台流处理异常: {e}")
            time.sleep(0.1)
def gen_frames():
    """生成视频流"""
    while True:
        with FRAME_LOCK:
            frame = LAST_PROCESSED_FRAME.copy() if LAST_PROCESSED_FRAME is not None else None

        if frame is None:
            time.sleep(0.03)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# 事件列表模拟数据
def generate_events():
    """
    获取真实的报警事件列表
    """
    events = []
    # 1. 从共享的全局 alert_manager 获取最近的历史记录
    # recent_alerts = alert_manager.get_recent_alerts(limit=10)
    # for alert in recent_alerts:
    #     video_url = ''
    #     if alert.video_clip_path and os.path.exists(alert.video_clip_path):
    #         # 如果报警片段保存在 local_videos 目录下，直接使用相对路径
    #         video_url = f'/video_clips/{os.path.basename(alert.video_clip_path)}'
    #     events.append({
    #         'type': alert.alert_type,
    #         'time': datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S"),
    #         'camera': 'Camera 01',
    #         'description': alert.description,
    #         'confidence': 85,
    #         'video_url': video_url
    #     })
    local_events = scan_local_videos()
    # 去重：若 video_url 已经存在于报警事件中，则不再重复添加
    existing_urls = {ev.get('video_url') for ev in events if ev.get('video_url')}
    for ev in local_events:
        if ev['video_url'] not in existing_urls:
            events.append(ev)
    # 如果历史记录为空，返回空列表或默认提示
    if not events:
        # 可选：返回一个默认的“无报警”状态，或者保持为空
        pass

    return events

# 初始化全局摄像头管理器（需在 detector/monitor 等初始化之后）
camera_manager = CameraManager(CAMERAS_CONFIG_FILE)

# 修改原有的全局 camera 变量为使用 manager 的当前实例
def get_current_camera():
    return camera_manager.current_camera
@app.route('/')
def index():
    """主页面"""
    # 更新模拟数据
    global alert_data
    alert_data['count'] += random.randint(-3, 3)
    alert_data['percentage'] = round(random.uniform(0.5, 2.0), 1)
    alert_data['in_zone_time'] = round(random.uniform(0.5, 1.5), 1)
    alert_data['motion_detected'] = random.choice([True, False])
    alert_data['targets'] = {
        'person': random.randint(1, 10),
        'vehicle': random.randint(0, 5),
        'bicycle': random.randint(0, 3)
    }

    events = generate_events()
    current_time = datetime.now()

    return render_template('index_pure.html',
                           alert_count=alert_data['count'],
                           percentage=alert_data['percentage'],
                           zone_time=alert_data['in_zone_time'],
                           targets=alert_data['targets'],
                           events=events,
                           datetime=datetime,
                           current_time=current_time)


@app.route('/video_feed')
def video_feed():
    """视频流路由"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/refresh_data')
def refresh_data():
    """手动刷新数据"""
    #ret = monitor.get_stats_overlay_dict()
    global alert_data
    # alert_data['count'] = ret["Total_Produced"]
    # alert_data['percentage'] = ret['Abnormal_Rate']
    # alert_data['in_zone_time'] = ret['avg_cycle_time']

    return jsonify(alert_data)


@app.route('/api/events')
def get_events():
    """获取事件列表"""
    events = generate_events()
    return jsonify(events)


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'ok',
        'camera_status': 'simulated' if camera_manager.current_camera.is_simulated else 'connected',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/video_clips/<filename>')
def serve_video_clip(filename):
    """提供本地异常视频文件，支持 Range 请求"""
    # 安全处理文件名
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(LOCAL_VIDEO_DIR, safe_filename)

    print(f"[视频请求] 文件名: {safe_filename}, 完整路径: {file_path}")  # 调试日志

    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        abort(404)

    # 获取 MIME 类型
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = 'video/mp4'  # 默认

    # 使用 send_file 支持 Range 请求（允许拖动进度条）
    try:
        return send_file(
            file_path,
            mimetype=mime_type,
            conditional=True,  # 支持 If-Modified-Since 和 Range
            etag=True,  # 支持缓存
            last_modified=os.path.getmtime(file_path)
        )
    except Exception as e:
        print(f"[错误] 发送文件失败: {e}")
        abort(500)

# 修改截图功能，返回更详细的信息
@app.route('/screenshot')
def screenshot():
    """截图功能"""
    try:
        frame = camera_manager.current_camera.get_frame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保screenshots目录存在
        if not os.path.exists('screenshots'):
            os.makedirs('screenshots')

        filename = f"screenshots/screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)

        return jsonify({
            'status': 'success',
            'message': f'截图已保存',
            'filename': filename,
            'timestamp': timestamp
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'截图失败: {str(e)}'
        }), 500


@app.route('/api/configure_camera', methods=['POST'])
def configure_camera():
    """
    配置并切换RTSP摄像头
    请求体JSON格式:
    {
        "ip": "10.59.77.176",
        "port": 554,
        "username": "admin",
        "password": "lsnotes@byd",
        "channel": "602"
    }
    """
    try:
        data = request.get_json()
        required_fields = ['ipAddress', 'port', 'username', 'password', 'channel','cameraName']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'message': f'缺少必要字段: {field}'}), 400

        ip = data['ipAddress']
        port = int(data['port'])
        username = data['username']
        password = data['password']
        channel = data['channel']
        camera_name = data['cameraName']

        # 对用户名和密码进行URL编码，防止特殊字符破坏URL结构
        encoded_username = urllib.parse.quote(username, safe='')
        encoded_password = urllib.parse.quote(password, safe='')

        # 格式: rtsp://username:password@ip:port/Streaming/Channels/channel
        rtsp_url = f"rtsp://{encoded_username}:{encoded_password}@{ip}:{port}/Streaming/Channels/{channel}"

        if camera_manager.active_camera_id is None:
            # 没有激活摄像头，则新增
            new_id = camera_manager.add_camera(camera_name, rtsp_url)
            success, msg = camera_manager.switch_to(new_id)
        else:
            # 更新当前激活的摄像头（覆盖原有配置）
            success, msg = camera_manager.reconfigure_active(rtsp_url, camera_name)
            if not success:
                # 如果更新失败，可以考虑作为新增
                new_id = camera_manager.add_camera(camera_name, rtsp_url)
                success, msg = camera_manager.switch_to(new_id)

        if success:
            # 返回成功并通知前端刷新
            return jsonify({'success': True, 'message': msg, 'rtsp_url': rtsp_url})
        else:
            return jsonify({'success': False, 'message': msg}), 400

    except Exception as e:
        print(f"配置摄像头异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'服务器错误: {str(e)}'}), 500

# 添加获取摄像机列表的API
@app.route('/api/cameras')
def get_cameras():
    return jsonify(camera_manager.get_list())

# 新增切换摄像头的接口
@app.route('/api/switch_camera', methods=['POST'])
def switch_camera():
    camera_id = request.json.get('id')
    print("切换到相机:",camera_id)
    if not camera_id:
        return jsonify({'success': False, 'message': '缺少摄像头ID'}), 400
    success, msg = camera_manager.switch_to(int(camera_id))
    if success:
        return jsonify({'success': True, 'message': msg})
    else:
        return jsonify({'success': False, 'message': msg}), 400

if __name__ == '__main__':
    print("=" * 50)
    print("SOP监控系统启动中...")
    print("=" * 50)

    # 创建截图目录
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # 初始化配置 & AI模型
    config = Config(source_url='rtsp://admin:lsnotes@byd@10.59.77.176:554/Streaming/Channels/602')
    # global detector, monitor
    # detector = YOLODetector(config)
    # monitor = SOPMonitor(config)

    # 初始化默认摄像头
    if not camera_manager.cameras:
        default_id = camera_manager.add_camera("默认摄像头", config.source_url)
        camera_manager.switch_to(default_id)
    else:
        camera_manager.switch_to(camera_manager.cameras[0]['id'])

    # 启动【后台统一推流+AI线程】守护线程
    stream_thread = threading.Thread(target=backend_stream_loop, daemon=True)
    stream_thread.start()
    print("✅ 后台视频流 & AI 推理线程已启动")

    print(f"🌐 访问地址: http://localhost:5000")
    print(f"📹 视频流地址: http://localhost:5000/video_feed")
    print("=" * 50)

    # Flask 开启多线程，支持多客户端并发
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
        #app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        # 程序退出时终止后台线程
        BACKEND_RUNNING = False
        if camera_manager.current_camera:
            camera_manager.current_camera.del_cap()
        print("程序已退出，资源已释放")
