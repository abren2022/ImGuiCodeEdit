from flask import Flask, render_template, Response, jsonify, url_for, request
import random
import os
import json
from datetime import datetime, timedelta
import cv2
import urllib.parse
import time
import threading
import gc
from AsseBehaviorMonitoring import Config,SOPMonitor,YOLODetector
from AsseBehaviorMonitoring import alert_manager
app = Flask(__name__)
alert_data = {
    'count': 522,
    'percentage': 1.1,
    'in_zone_time': 0.8,
    'motion_detected': False,
    'targets': {'person': 5, 'vehicle': 3, 'bicycle': 1}
}

CAMERAS_CONFIG_FILE = 'cameras.json'

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
            self.current_camera.release()
        self.current_camera = VideoCamera(cam_info['rtsp'])
        self.active_camera_id = camera_id
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
        if self.active_camera_id == camera_id:
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
        self.is_simulated = False
        self.lock = threading.RLock()
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
        if self.cap:
            self.cap.release()
            self.cap = None
        # 强制垃圾回收，帮助 FFmpeg 清理内部资源
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
        if not self.is_simulated and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_simulated = True


def gen_frames():
    """生成视频流"""
    while True:
        try:
            frame = camera_manager.get_active_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            detections = detector.detect_and_track(frame)
            current_time = time.time()
            monitor.update_all_workpieces(detections, current_time)
            frame = monitor.draw_status(frame, detections)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"视频流错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

# 事件列表模拟数据
def generate_events():
    """
    获取真实的报警事件列表
    """
    events = []
    # 1. 从共享的全局 alert_manager 获取最近的历史记录
    recent_alerts = alert_manager.get_recent_alerts(limit=10)

    for alert in recent_alerts:
        # 2. 格式化数据以匹配前端需要的格式
        events.append({
            'type': alert.alert_type,   # 例如: 'timeout', 'missing_step'
            'time': datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S"),
            'camera': 'Camera 01',  # 如果有多个摄像头，这里需要动态获取
            'description': alert.description,  # 额外字段，前端可选显示
            'video_url':alert.video_clip_path if alert.video_clip_path else ''  # 如果有视频片段路径
        })

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
    ret = monitor.get_stats_overlay_dict()
    global alert_data
    alert_data['count'] = ret["Total_Produced"]
    alert_data['percentage'] = ret['Abnormal_Rate']
    alert_data['in_zone_time'] = ret['avg_cycle_time']

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
        'camera_status': 'simulated' if camera.is_simulated else 'connected',
        'timestamp': datetime.now().isoformat()
    })


# 修改截图功能，返回更详细的信息
@app.route('/screenshot')
def screenshot():
    """截图功能"""
    try:
        frame = camera.get_frame()
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
    if not camera_id:
        return jsonify({'success': False, 'message': '缺少摄像头ID'}), 400
    success, msg = camera_manager.switch_to(int(camera_id))
    if success:
        return jsonify({'success': True, 'message': msg})
    else:
        return jsonify({'success': False, 'message': msg}), 400

if __name__ == '__main__':
    # 创建screenshots目录

    # 初始化摄像头
    print("=" * 50)
    print("SOP监控系统启动中...")
    print("=" * 50)
    config = Config(source_url='rtsp://admin:lsnotes@byd@10.59.77.176:554/Streaming/Channels/602')
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    if not camera_manager.cameras:
        # 从 config 中创建默认摄像头
        default_id = camera_manager.add_camera("默认摄像头", config.source_url)
        camera_manager.switch_to(default_id)
    else:
        # 激活第一个摄像头
        camera_manager.switch_to(camera_manager.cameras[0]['id'])
    detector = YOLODetector(config)
    monitor = SOPMonitor(config)

    print(f"🌐 访问地址: http://localhost:5000")
    print(f"📹 视频流地址: http://localhost:5000/video_feed")
    print("=" * 50)

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)



    <!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI天眼监控智能体</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%234169e1'/><circle cx='50' cy='50' r='25' fill='white'/><circle cx='50' cy='50' r='10' fill='%230a0e27'/></svg>">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg-color: #0a0e27;
            --sidebar-color: #1a1f3a;
            --card-bg: #131833;
            --accent-blue: #4169e1;
            --accent-cyan: #00d4ff;
            --text-white: #ffffff;
            --text-gray: #8a8d9f;
            --alert-red: #ff4757;
            --warning-orange: #ffa502;
            --success-green: #2ed573;
            --hover-bg: #252b4a;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Arial', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-white);
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* 侧边栏样式 */
        .sidebar {
            width: 260px;
            background-color: var(--sidebar-color);
            padding: 20px 0;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow-y: auto;
        }

        .logo-section {
            display: flex;
            align-items: center;
            padding: 0 20px;
            margin-bottom: 20px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background-color: var(--accent-blue);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: bold;
            margin-right: 10px;
        }

        .logo-text {
            font-size: 16px;
            font-weight: bold;
        }

        .divider {
            height: 1px;
            background-color: #2a2f4a;
            margin: 10px 20px;
        }

        .menu {
            flex: 1;
        }

        .menu-item {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            color: var(--text-gray);
            text-decoration: none;
            transition: all 0.3s;
            cursor: pointer;
        }

        .menu-item:hover {
            background-color: var(--hover-bg);
            color: var(--text-white);
        }

        .menu-item.active {
            background-color: var(--hover-bg);
            color: var(--text-white);
            border-right: 3px solid var(--accent-cyan);
        }

        .menu-item .icon {
            margin-right: 10px;
            font-size: 16px;
        }

        .camera-section {
            padding: 20px;
        }

        .camera-section h3 {
            color: var(--text-gray);
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .camera-item {
            display: flex;
            align-items: center;
            padding: 8px 5px;
            color: var(--text-gray);
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .camera-item:hover {
            color: var(--text-white);
            background-color: var(--hover-bg);
            border-radius: 6px;
            padding-left: 10px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-dot.online {
            background-color: var(--success-green);
            box-shadow: 0 0 6px var(--success-green);
        }

        .status-dot.offline {
            background-color: var(--alert-red);
        }

        .camera-icon {
            margin-right: 5px;
        }

        .search-box {
            padding: 20px;
        }

        .search-box input {
            width: 100%;
            padding: 8px 12px;
            background-color: #0f1535;
            border: 1px solid #2a2f4a;
            color: var(--text-white);
            border-radius: 6px;
            outline: none;
        }

        .search-box input:focus {
            border-color: var(--accent-cyan);
        }

        /* 主内容区 */
        .main-content {
            flex: 1;
            display: flex;
            gap: 15px;
            padding: 20px;
            overflow: hidden;
        }

        .left-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow: hidden;
        }

        .right-panel {
            flex: 1;
            background-color: var(--card-bg);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }

        .right-panel-header {
            padding: 15px 20px;
            border-bottom: 1px solid #2a2f4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }

        .right-panel-header h3 {
            font-size: 16px;
            font-weight: bold;
        }

        .anomaly-list-container {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }

        .anomaly-list {
            padding: 10px;
        }

        .anomaly-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            margin-bottom: 12px;
            padding: 12px;
            display: flex;
            gap: 12px;
            transition: transform 0.2s, background-color 0.2s;
            cursor: pointer;
            border-left: 3px solid var(--accent-cyan);
        }

        .anomaly-card:hover {
            background-color: rgba(255, 255, 255, 0.1);
            transform: translateX(3px);
        }

        .thumbnail {
            width: 80px;
            height: 60px;
            background: linear-gradient(135deg, #1a1f3a, #0a0e27);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            flex-shrink: 0;
        }

        .anomaly-info {
            flex: 1;
            min-width: 0;
        }

        .anomaly-title {
            font-size: 14px;
            font-weight: bold;
            text-transform: capitalize;
            margin-bottom: 4px;
        }

        .anomaly-meta {
            font-size: 11px;
            color: var(--text-gray);
            display: flex;
            gap: 12px;
            margin-bottom: 4px;
        }

        .anomaly-time {
            font-size: 11px;
            color: var(--text-gray);
        }

        .confidence {
            display: inline-block;
            background-color: rgba(0, 212, 255, 0.2);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            color: var(--accent-cyan);
        }

        .anomaly-stats-card {
            background-color: rgba(0, 212, 255, 0.05);
            border-bottom: 1px solid #2a2f4a;
            padding: 12px 15px;
            flex-shrink: 0;
        }

        .stats-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .stats-header span {
            font-size: 12px;
            color: var(--text-gray);
        }

        .date-tabs {
            display: flex;
            gap: 6px;
        }

        .date-tab {
            background: transparent;
            border: 1px solid #2a2f4a;
            color: var(--text-gray);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .date-tab:hover {
            background: var(--hover-bg);
            border-color: var(--accent-cyan);
        }

        .date-tab.active {
            background: var(--accent-cyan);
            border-color: var(--accent-cyan);
            color: var(--bg-color);
        }

        .anomaly-count {
            font-size: 28px;
            font-weight: bold;
            color: var(--accent-cyan);
            margin-top: 5px;
        }

        .anomaly-count-label {
            font-size: 11px;
            color: var(--text-gray);
        }

        .video-section {
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            flex-shrink: 0;
        }

        .video-container {
            position: relative;
            aspect-ratio: 16/9;
            width: 100%;
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .video-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            padding: 10px 15px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            pointer-events: none;
        }

        .overlay-top {
            display: flex;
            justify-content: space-between;
        }

        .overlay-top span {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }

        .overlay-bottom {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
        }

        .ai-badge {
            background-color: rgba(0, 255, 0, 0.2);
            color: var(--success-green);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            border: 1px solid var(--success-green);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            flex-shrink: 0;
        }

        .stat-card {
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            transition: transform 0.3s;
        }

        .stat-card:hover {
            transform: translateY(-2px);
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-cyan);
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 11px;
            color: var(--text-gray);
            margin-bottom: 5px;
        }

        .stat-change {
            font-size: 11px;
            font-weight: bold;
        }

        .stat-change.up {
            color: var(--success-green);
        }

        .stat-change.warning {
            color: var(--warning-orange);
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .header-left h1 {
            font-size: 20px;
            font-weight: bold;
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 5px;
            background-color: rgba(255, 71, 87, 0.1);
            padding: 4px 8px;
            border-radius: 4px;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background-color: var(--alert-red);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .live-text {
            color: var(--alert-red);
            font-size: 11px;
            font-weight: bold;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .header-right {
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 6px 12px;
            background-color: #1a1f3a;
            color: var(--text-white);
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: background-color 0.3s;
        }

        .btn:hover {
            background-color: var(--hover-bg);
        }

        .info-banner {
            background-color: rgba(0, 212, 255, 0.1);
            border: 1px solid var(--accent-cyan);
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 15px;
            font-size: 12px;
            color: var(--accent-cyan);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* 滚动条 */
        .anomaly-list-container::-webkit-scrollbar,
        .sidebar::-webkit-scrollbar {
            width: 5px;
        }

        .anomaly-list-container::-webkit-scrollbar-track,
        .sidebar::-webkit-scrollbar-track {
            background: var(--card-bg);
        }

        .anomaly-list-container::-webkit-scrollbar-thumb,
        .sidebar::-webkit-scrollbar-thumb {
            background: var(--accent-cyan);
            border-radius: 3px;
        }

        /* 模态框样式 */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            z-index: 10000;
            justify-content: center;
            align-items: center;
        }

        .modal-content {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 24px;
            width: 500px;
            max-width: 90%;
            border: 1px solid var(--accent-cyan);
        }

        .modal-content h4 {
            margin-bottom: 20px;
            color: var(--accent-cyan);
        }

        .modal-content .form-group {
            margin-bottom: 15px;
        }

        .modal-content label {
            display: block;
            margin-bottom: 6px;
            font-size: 13px;
            color: var(--text-gray);
        }

        .modal-content input,
        .modal-content select {
            width: 100%;
            padding: 8px 12px;
            background-color: #0f1535;
            border: 1px solid #2a2f4a;
            color: var(--text-white);
            border-radius: 6px;
            outline: none;
            font-size: 14px;
        }

        .modal-content input:focus {
            border-color: var(--accent-cyan);
        }

        .modal-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 12px;
            margin-top: 20px;
        }

        .modal-buttons button {
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            border: none;
        }

        .btn-confirm {
            background-color: var(--accent-cyan);
            color: var(--bg-color);
        }

        .btn-cancel {
            background-color: #2a2f4a;
            color: var(--text-white);
        }

        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }

        @media (max-width: 1000px) {
            .right-panel { display: none; }
            .left-panel { flex: 1; }
        }

        @media (max-width: 768px) {
            .sidebar { width: 60px; }
            .logo-text, .menu-item span:not(.icon), .camera-section h3, .camera-item span:not(.status-dot):not(.camera-icon), .search-box { display: none; }
            .menu-item { justify-content: center; padding: 12px; }
            .camera-item { justify-content: center; }
            .stats-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
        <div class="logo-section">
            <div class="logo-icon">D</div>
            <div class="logo-text">天眼监控</div>
        </div>
        <div class="divider"></div>
        <nav class="menu">
            <div class="menu-item active" data-page="monitor">
                <span class="icon">📊</span>
                <span>实时监控</span>
            </div>
            <div class="menu-item" id="cameraConfigBtn">
                <span class="icon">📷</span>
                <span>点位相机配置</span>
            </div>
            <div class="menu-item" data-page="workflow">
                <span class="icon">🔄</span>
                <span>工作流</span>
            </div>
            <div class="menu-item" data-page="settings">
                <span class="icon">⚙️</span>
                <span>系统设置</span>
            </div>
        </nav>
        <div class="camera-section">
            <h3>摄像机列表</h3>
            <div id="cameraListContainer"></div>
        </div>
        <div class="search-box">
            <input type="text" placeholder="搜索事件..." id="searchInput">
        </div>
    </aside>

    <!-- 主内容 -->
    <main class="main-content">
        <div class="left-panel">
            <div class="header">
                <div class="header-left">
                    <h1>实时监控</h1>
                    <div class="live-indicator">
                        <span class="live-dot"></span>
                        <span class="live-text">LIVE</span>
                    </div>
                </div>
                <div class="header-right">
                    <button class="btn" id="screenshotBtn">📸 截图</button>
                    <button class="btn" id="fullscreenBtn">⛶ 全屏</button>
                    <button class="btn" id="refreshDataBtn">🔄 刷新</button>
                </div>
            </div>
            <div class="info-banner">
                <span>ℹ️ 自动刷新 | 当前: <span id="currentTime"></span></span>
                <span id="connectionStatus" style="color: var(--success-green);">● 已连接</span>
            </div>
            <div class="video-section">
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" alt="监控视频流" id="videoFeed">
                    <div class="video-overlay">
                        <div class="overlay-top">
                            <span id="cameraNameLabel"></span>
                            <span>1920x1080 @ 30FPS</span>
                        </div>
                        <div class="overlay-bottom">
                            <div><span class="ai-badge">🤖 AI Detection Active</span></div>
                            <span id="videoTime"></span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="alertCount">0</div>
                    <div class="stat-label">当前生成总量</div>
                    <div class="stat-change warning" id="percentage"></div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="zoneTime">0s</div>
                    <div class="stat-label">平均单位生产耗时</div>
                    <div class="stat-change info">今日事件</div>
                </div>
            </div>
        </div>
        <div class="right-panel">
            <div class="right-panel-header">
                <h3>🚨 异常片段</h3>
                <button class="btn" id="refreshEventsBtn" style="padding: 4px 8px;">🔄</button>
            </div>
            <div class="anomaly-stats-card">
                <div class="stats-header">
                    <span>异常次数统计</span>
                    <div class="date-tabs">
                        <button class="date-tab active" data-range="today">今日</button>
                        <button class="date-tab" data-range="yesterday">昨日</button>
                        <button class="date-tab" data-range="week">本周</button>
                    </div>
                </div>
                <div class="anomaly-count" id="anomalyCount">0</div>
                <div class="anomaly-count-label">次异常事件</div>
            </div>
            <div class="anomaly-list-container">
                <div class="anomaly-list" id="anomalyList">
                    <div style="text-align: center; color: var(--text-gray); padding: 20px;">加载中...</div>
                </div>
            </div>
        </div>
    </main>
</div>

<!-- 异常事件详情模态框 -->
<div id="eventModal" class="modal">
    <div class="modal-content">
        <h4 id="modalTitle">异常事件详情</h4>
        <div id="modalVideoContainer" style="margin-top: 10px;">
            <video id="eventVideo" controls style="width: 100%; max-height: 300px; display: none;">
                您的浏览器不支持视频播放。
            </video>
        </div>
        <p id="modalDetail" style="margin-top: 10px;"></p>
        <div style="text-align: right; margin-top: 15px;">
            <button class="btn-cancel" id="closeModal">关闭</button>
        </div>
    </div>
</div>

<!-- 点位相机配置模态框 -->
<div id="cameraConfigModal" class="modal">
    <div class="modal-content">
        <h4>📹 点位相机配置</h4>
        <form id="cameraConfigForm">
            <div class="form-group">
                <label>点位名称（可选）</label>
                <input type="text" id="cameraName" placeholder="">
            </div>
            <div class="form-group">
                <label>IP地址 *</label>
                <input type="text" id="ipAddress" placeholder="10.59.77.176" required>
            </div>
            <div class="form-group">
                <label>端口 *</label>
                <input type="number" id="port" placeholder="554" value="554" required>
            </div>
            <div class="form-group">
                <label>用户名 *</label>
                <input type="text" id="username" placeholder="admin" required>
            </div>
            <div class="form-group">
                <label>密码 *</label>
                <input type="password" id="password" placeholder="密码" required>
            </div>
            <div class="form-group">
                <label>码流编号 (通道) *</label>
                <input type="text" id="channel" placeholder="602" required>
            </div>
            <div class="modal-buttons">
                <button type="button" class="btn-cancel" id="cancelConfigBtn">取消</button>
                <button type="submit" class="btn-confirm">确认添加/切换</button>
            </div>
        </form>
    </div>
</div>

<script>
    // ======================== 全局变量 ========================
    let refreshInterval, eventsInterval;
    let allEvents = [];
    let currentDateRange = 'today';

    // ======================== 辅助函数 ========================
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed; top: 20px; right: 20px; padding: 10px 16px;
            background-color: ${type === 'success' ? '#2ed573' : type === 'error' ? '#ff4757' : '#00d4ff'};
            color: white; border-radius: 6px; font-size: 13px; z-index: 9999;
            animation: slideIn 0.3s ease; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 2500);
        }, 2500);
    }

    function updateTime() {
        const now = new Date();
        const timeStr = now.toISOString().slice(0, 19).replace('T', ' ');
        document.getElementById('currentTime').innerText = timeStr;
        document.getElementById('videoTime').innerText = timeStr;
    }

    // 判断事件是否在日期范围内（基于事件时间字符串 HH:MM:SS，假设发生在今天）
    function isEventInRange(eventTimeStr, range) {
        const now = new Date();
        const [hours, minutes, seconds] = eventTimeStr.split(':').map(Number);
        const eventDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hours, minutes, seconds);
        const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterdayStart = new Date(todayStart);
        yesterdayStart.setDate(yesterdayStart.getDate() - 1);
        const weekStart = new Date(todayStart);
        weekStart.setDate(weekStart.getDate() - now.getDay());
        if (range === 'today') return eventDate >= todayStart;
        if (range === 'yesterday') return eventDate >= yesterdayStart && eventDate < todayStart;
        if (range === 'week') return eventDate >= weekStart;
        return false;
    }

    function updateAnomalyCount() {
        let filtered = allEvents;
        if (currentDateRange === 'today') filtered = allEvents.filter(ev => isEventInRange(ev.time, 'today'));
        else if (currentDateRange === 'yesterday') filtered = allEvents.filter(ev => isEventInRange(ev.time, 'yesterday'));
        else if (currentDateRange === 'week') filtered = allEvents.filter(ev => isEventInRange(ev.time, 'week'));
        document.getElementById('anomalyCount').innerText = filtered.length;
    }

    // 刷新异常事件列表
    function refreshAnomalyList() {
        fetch('/api/events')
            .then(res => res.json())
            .then(data => {
                allEvents = data;
                const container = document.getElementById('anomalyList');
                if (!container) return;
                if (allEvents.length === 0) {
                    container.innerHTML = '<div style="text-align: center; color: var(--text-gray); padding: 20px;">暂无异常事件</div>';
                    updateAnomalyCount();
                    return;
                }
                container.innerHTML = '';
                allEvents.forEach(event => {
                    const card = document.createElement('div');
                    card.className = `anomaly-card priority-${event.priority || 'medium'}`;
                    let icon = '🎥';
                    if (event.type === 'person') icon = '👤';
                    else if (event.type === 'vehicle') icon = '🚗';
                    else if (event.type === 'bicycle') icon = '🚲';
                    else if (event.type === 'motion') icon = '🎬';
                    card.innerHTML = `
                        <div class="thumbnail"><div class="thumbnail-icon">${icon}</div></div>
                        <div class="anomaly-info">
                            <div class="anomaly-title">${event.type}</div>
                            <div class="anomaly-meta"><span>${event.camera}</span><span class="confidence">${event.confidence || 85}%</span></div>
                            <div class="anomaly-time">${event.time}</div>
                        </div>
                    `;
                    card.dataset.event = JSON.stringify(event);
                    card.addEventListener('click', (e) => {
                        e.stopPropagation();
                        const evData = JSON.parse(card.dataset.event);
                        showEventModal(evData);
                    });
                    container.appendChild(card);
                });
                updateAnomalyCount();
            })
            .catch(err => console.error('刷新异常列表失败:', err));
    }

    function showEventModal(event) {
        const modal = document.getElementById('eventModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalDetail = document.getElementById('modalDetail');
        const videoEl = document.getElementById('eventVideo');
        const videoContainer = document.getElementById('modalVideoContainer');

        modalTitle.innerText = `${event.type} 异常片段`;
        modalDetail.innerHTML = `摄像机: ${event.camera}<br>时间: ${event.time}<br>置信度: ${event.confidence || 85}%`;

        // 清除旧的加载状态
        if (videoEl) {
            videoEl.pause();
            videoEl.src = '';
            videoEl.removeAttribute('src');
            videoEl.load();
        }
        // 移除已有的加载提示元素（如果存在）
        const oldMsg = document.getElementById('videoErrorMsg');
        if (oldMsg) oldMsg.remove();

        if (event.video_url && event.video_url.trim() !== '') {
            videoEl.style.display = 'block';
            // 显示加载中（可选）
            videoEl.style.opacity = '0.5';
            // 设置视频源并尝试加载
            videoEl.src = event.video_url;
            videoEl.load();

            // 监听加载成功
            videoEl.oncanplay = () => {
                videoEl.style.opacity = '1';
                const errMsg = document.getElementById('videoErrorMsg');
                if (errMsg) errMsg.remove();
            };
            // 监听错误
            videoEl.onerror = (e) => {
                console.error('视频加载失败:', e);
                videoEl.style.display = 'none';
                const errorDiv = document.createElement('div');
                errorDiv.id = 'videoErrorMsg';
                errorDiv.style.color = '#ffa502';
                errorDiv.style.marginTop = '10px';
                errorDiv.innerText = '⚠️ 视频无法播放，可能文件缺失或格式不支持。';
                videoContainer.appendChild(errorDiv);
            };
        } else {
            videoEl.style.display = 'none';
            videoEl.src = '';
            modalDetail.innerHTML += '<br><span style="color: #ffa502;">无视频片段</span>';
        }

        modal.style.display = 'flex';
    }

    function closeEventModal() {
        const modal = document.getElementById('eventModal');
        const videoEl = document.getElementById('eventVideo');
        if (videoEl) {
            videoEl.pause();
            videoEl.src = '';   // 释放资源
        }
        modal.style.display = 'none';
    }
    function closeEventModal() {
        document.getElementById('eventModal').style.display = 'none';
    }

    // 刷新左侧统计卡片
    function updateStats() {
        fetch('/refresh_data')
            .then(res => res.json())
            .then(data => {
                document.getElementById('alertCount').innerText = data.count;
                document.getElementById('zoneTime').innerText = data.in_zone_time;
                document.getElementById('percentageChange').innerHTML = `↑ ${data.percentage}`;
            })
            .catch(err => console.error('更新统计失败:', err));
    }

    // 截图
    function takeScreenshot() {
        const btn = document.getElementById('screenshotBtn');
        const orig = btn.innerHTML;
        btn.innerHTML = '💾 保存...';
        btn.disabled = true;
        fetch('/screenshot')
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') showNotification('✅ 截图已保存', 'success');
                else showNotification('❌ 截图失败: ' + data.message, 'error');
            })
            .catch(() => showNotification('❌ 截图失败', 'error'))
            .finally(() => {
                btn.innerHTML = orig;
                btn.disabled = false;
            });
    }

    function toggleFullscreen() {
        const container = document.querySelector('.video-container');
        if (!document.fullscreenElement) {
            container.requestFullscreen().catch(err => showNotification('全屏失败', 'error'));
        } else {
            document.exitFullscreen();
        }
    }

    // 切换摄像头（通过后端接口）
    function switchToCamera(cameraId, cameraName) {
        fetch('/api/switch_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: cameraId })
        })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    showNotification(`已切换到 ${cameraName}`, 'success');
                    // 刷新视频流（加时间戳防止缓存）
                    //const videoImg = document.getElementById('videoFeed');
                    //videoImg.src = '/video_feed?'+ Date.now();
                    document.getElementById('cameraNameLabel').innerHTML = `📹 ${cameraName}`;
                } else {
                    showNotification('切换失败: ' + data.message, 'error');
                }
            })
            .catch(err => showNotification('网络错误', 'error'));
    }

    // 加载相机列表（动态渲染侧边栏）
    function loadCameraList() {
        fetch('/api/cameras')
            .then(res => res.json())
            .then(cameras => {
                const container = document.getElementById('cameraListContainer');
                if (!container) return;
                container.innerHTML = '';
                if (cameras.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-gray); padding: 8px;">暂无摄像头，请点击“点位相机配置”添加</div>';
                    return;
                }
                cameras.forEach(cam => {
                    const div = document.createElement('div');
                    div.className = 'camera-item';
                    div.dataset.camera = cam.id;
                    div.innerHTML = `
                        <span class="status-dot ${cam.status}"></span>
                        <span class="camera-icon">📹</span>
                        <span>${cam.name}</span>
                    `;
                    div.addEventListener('click', () => {
                        if (cam.status !== 'online') {
                            showNotification('摄像头离线', 'error');
                            return;
                        }
                        switchToCamera(cam.id, cam.name);
                    });
                    container.appendChild(div);
                });
            })
            .catch(err => console.error('加载摄像机列表失败:', err));
    }

    // 点位相机配置对话框逻辑
    function showCameraConfigModal() {
        document.getElementById('cameraConfigModal').style.display = 'flex';
    }

    function closeCameraConfigModal() {
        document.getElementById('cameraConfigModal').style.display = 'none';
        document.getElementById('cameraConfigForm').reset();
    }

    async function submitCameraConfig(event) {
        event.preventDefault();
        const name = document.getElementById('cameraName').value.trim() || '未命名相机';
        const ip = document.getElementById('ipAddress').value.trim();
        const port = parseInt(document.getElementById('port').value);
        const username = document.getElementById('username').value.trim();
        const password = document.getElementById('password').value;
        const channel = document.getElementById('channel').value.trim();

        if (!ip || !port || !username || !password || !channel) {
            showNotification('请填写所有带*的字段', 'error');
            return;
        }
eventModal
        const configData = { name, ip, port, username, password, channel };
        const submitBtn = document.querySelector('#cameraConfigForm .btn-confirm');
        const originalText = submitBtn.innerText;
        submitBtn.innerText = '配置中...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/configure_camera', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(configData)
            });
            const result = await response.json();
            if (result.success) {
                showNotification('✅ 摄像头配置成功，正在切换', 'success');
                closeCameraConfigModal();
                // 重新加载相机列表
                loadCameraList();
                // 刷新视频流
                const videoImg = document.getElementById('videoFeed');
                videoImg.src = '/video_feed?';
                // 更新摄像头名称显示
                document.getElementById('cameraNameLabel').innerHTML = `📹 ${name}`;
            } else {
                showNotification(`❌ 配置失败: ${result.message}`, 'error');
            }
        } catch (error) {
            console.error('配置摄像头错误:', error);
            showNotification('网络错误，请检查后端服务', 'error');
        } finally {
            submitBtn.innerText = originalText;
            submitBtn.disabled = false;
        }
    }

    // 连接状态检测
    let reconnectAttempts = 0;
    function checkVideoConnection() {
        const videoImg = document.getElementById('videoFeed');
        if (!videoImg) return;
        const img = new Image();
        img.onload = () => {
            reconnectAttempts = 0;
            document.getElementById('connectionStatus').innerHTML = '● 已连接';
            document.getElementById('connectionStatus').style.color = 'var(--success-green)';
        };
        img.onerror = () => {
            if (reconnectAttempts < 3) {
                reconnectAttempts++;
                const status = document.getElementById('connectionStatus');
                status.innerHTML = `● 重连 (${reconnectAttempts}/3)`;
                status.style.color = 'var(--warning-orange)';
                setTimeout(() => {
                    videoImg.src = videoImg.src.split('?')[0] + `?t=${Date.now()}`;
                }, 2000);
            } else {
                document.getElementById('connectionStatus').innerHTML = '● 连接失败';
                document.getElementById('connectionStatus').style.color = 'var(--alert-red)';
            }
        };
        img.src = videoImg.src;
    }

    // 搜索事件过滤
    function filterAnomalies() {
        const keyword = document.getElementById('searchInput').value.toLowerCase();
        const cards = document.querySelectorAll('.anomaly-card');
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(keyword) ? 'flex' : 'none';
        });
    }

    // ======================== 初始化所有事件 ========================
    function init() {
        // 更新时间显示
        updateTime();
        setInterval(updateTime, 1000);

        // 加载相机列表
        loadCameraList();

        // 加载异常事件列表
        refreshAnomalyList();

        // 加载统计数据
        updateStats();

        // 定时刷新
        refreshInterval = setInterval(updateStats, 5000);
        eventsInterval = setInterval(refreshAnomalyList, 10000);
        setInterval(checkVideoConnection, 30000);

        // 按钮事件绑定
        const screenshotBtn = document.getElementById('screenshotBtn');
        if (screenshotBtn) screenshotBtn.addEventListener('click', takeScreenshot);

        const fullscreenBtn = document.getElementById('fullscreenBtn');
        if (fullscreenBtn) fullscreenBtn.addEventListener('click', toggleFullscreen);

        const refreshDataBtn = document.getElementById('refreshDataBtn');
        if (refreshDataBtn) refreshDataBtn.addEventListener('click', () => {
            updateStats();
            showNotification('数据已刷新', 'success');
        });

        const refreshEventsBtn = document.getElementById('refreshEventsBtn');
        if (refreshEventsBtn) refreshEventsBtn.addEventListener('click', refreshAnomalyList);

        const searchInput = document.getElementById('searchInput');
        if (searchInput) searchInput.addEventListener('input', filterAnomalies);

        const closeModalBtn = document.getElementById('closeModal');
        if (closeModalBtn) closeModalBtn.addEventListener('click', closeEventModal);

        // 点位相机配置相关
        const configBtn = document.getElementById('cameraConfigBtn');
        if (configBtn) configBtn.addEventListener('click', showCameraConfigModal);

        const cancelConfigBtn = document.getElementById('cancelConfigBtn');
        if (cancelConfigBtn) cancelConfigBtn.addEventListener('click', closeCameraConfigModal);

        const configForm = document.getElementById('cameraConfigForm');
        if (configForm) configForm.addEventListener('submit', submitCameraConfig);

        // 日期切换标签
        document.querySelectorAll('.date-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('.date-tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                currentDateRange = this.dataset.range;
                updateAnomalyCount();
            });
        });

        // 其他菜单项点击提示（开发中）
        document.querySelectorAll('.menu-item[data-page]').forEach(item => {
            item.addEventListener('click', () => {
                if (!item.classList.contains('active')) {
                    showNotification('功能开发中', 'info');
                }
            });
        });

        // 页面可见性变化时调整刷新频率
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                clearInterval(refreshInterval);
                clearInterval(eventsInterval);
                refreshInterval = setInterval(updateStats, 15000);
                eventsInterval = setInterval(refreshAnomalyList, 30000);
            } else {
                clearInterval(refreshInterval);
                clearInterval(eventsInterval);
                refreshInterval = setInterval(updateStats, 5000);
                eventsInterval = setInterval(refreshAnomalyList, 10000);
                updateStats();
                refreshAnomalyList();
                checkVideoConnection();
            }
        });

        console.log('天眼监控系统已启动 - 所有按钮事件已绑定');
    }

    // 启动
    document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
