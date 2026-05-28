from flask import Flask, render_template, Response, jsonify, url_for
import random
import os
from datetime import datetime, timedelta
import cv2
import numpy as np
import time
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

class VideoCamera:
    def __init__(self, camera_id):
        self.video_source = camera_id
        self.cap = None
        self.is_simulated = False
        self.initialize_camera()

    def initialize_camera(self):
        """初始化摄像头"""
        try:
            print(f"正在尝试打开摄像头 {self.video_source}...")
            self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                print(f"摄像头 {self.video_source} 不可用，将使用模拟视频")
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"摄像头 {self.video_source} 无法读取画面，将使用模拟视频")
                    self.is_simulated = True
                else:
                    print(f"摄像头 {self.video_source} 连接成功！")
                    self.is_simulated = False

        except Exception as e:
            print(f"摄像头 {self.video_source} 初始化失败: {e}")
            print("将使用模拟视频")
            self.is_simulated = True

    def get_frame(self):
        """获取视频帧"""
        if not self.is_simulated and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                detections = detector.detect_and_track(frame)
                current_time = time.time()
                monitor.update_all_workpieces(detections,current_time)
                return monitor.draw_status(frame, detections)

        return None

    def __del__(self):
        if self.cap:
            self.cap.release()


def gen_frames():
    """生成视频流"""
    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
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
    ret = monitor.get_stats_overlay_text()
    print(ret)
    global alert_data
    alert_data['count'] += random.randint(-2, 2)
    alert_data['percentage'] = round(random.uniform(0.5, 2.0), 1)
    alert_data['in_zone_time'] = round(random.uniform(0.5, 1.5), 1)

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


# 添加获取摄像机列表的API
@app.route('/api/cameras')
def get_cameras():
    """获取摄像机列表"""
    cameras = []
    for i in range(1, 5):
        cameras.append({
            'id': i,
            'name': f'Camera {i:02d}',
            'status': 'online' if i != 3 else 'offline',  # 示例中摄像头3为离线
            'url': f'/video_feed?camera={i}'
        })
    return jsonify(cameras)

if __name__ == '__main__':
    # 创建screenshots目录
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # 初始化摄像头
    print("=" * 50)
    print("SOP监控系统启动中...")
    print("=" * 50)
    config = Config(source_url='rtsp://admin:lsnotes@byd@10.59.77.176:554/Streaming/Channels/602')
    detector = YOLODetector(config)
    monitor = SOPMonitor(config)
    camera = VideoCamera(config.source_url)

    if camera.is_simulated:
        print("ℹ️  使用模拟视频模式")
    else:
        print("✅ 摄像头连接成功")

    print(f"🌐 访问地址: http://localhost:5000")
    print(f"📹 视频流地址: http://localhost:5000/video_feed")
    print("=" * 50)

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)   这是网页 <!DOCTYPE html>
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
            width: 220px;
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

        /* 主内容区 - 左右两列布局 */
        .main-content {
            flex: 1;
            display: flex;
            gap: 15px;
            padding: 20px;
            overflow: hidden;
        }

        /* 左侧区域：视频 + 统计卡片 */
        .left-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 15px;
            overflow: hidden;
        }

        /* 右侧区域：异常片段列表 + 底部统计 */
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

        /* 异常列表容器 - 可滚动，与视频窗口高度对齐 */
        .anomaly-list-container {
            flex: 1;
            overflow-y: auto;
            min-height: 0;
        }

        .anomaly-list {
            padding: 10px;
        }

        /* 异常片段卡片 */
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

        .anomaly-card.priority-high {
            border-left-color: var(--alert-red);
        }

        .anomaly-card.priority-medium {
            border-left-color: var(--warning-orange);
        }

        .anomaly-card.priority-low {
            border-left-color: var(--success-green);
        }

        /* 缩略图占位 */
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
            overflow: hidden;
        }

        .thumbnail:hover .thumbnail-overlay {
        opacity: 1;
        }

        .thumbnail-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.3s;
        }

        .thumbnail img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        }

        .play-icon {
        color: white;
        font-size: 24px;
        text-shadow: 0 0 4px rgba(0,0,0,0.5);
        }

        .thumbnail-icon {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        color: #666;
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

        /* 底部统计卡片（异常次数） */
        .anomaly-stats-card {
            background-color: rgba(0, 212, 255, 0.05);
            border-top: none;
            border-bottom: 1px solid #2a2f4a;
            padding: 12px 15px;
            flex-shrink: 0; /* 保持不压缩 */
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

        /* 视频区域 */
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

        /* 统计卡片（左侧） */
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

        /* 头部按钮 */
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

        /* 弹窗样式 */
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
            padding: 20px;
            width: 500px;
            max-width: 90%;
            border: 1px solid var(--accent-cyan);
        }
        .modal-content h4 {
            margin-bottom: 10px;
        }
        .modal-content button {
            margin-top: 15px;
            background: var(--accent-cyan);
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
        }

        /* 响应式 */
        @media (max-width: 1000px) {
            .right-panel {
                display: none;
            }
            .left-panel {
                flex: 1;
            }
        }

        @media (max-width: 768px) {
            .sidebar {
                width: 60px;
            }
            .logo-text, .menu-item span:not(.icon),
            .camera-section h3, .camera-item span:not(.status-dot):not(.camera-icon),
            .search-box {
                display: none;
            }
            .menu-item {
                justify-content: center;
                padding: 12px;
            }
            .camera-item {
                justify-content: center;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 左侧边栏 (保持不变) -->
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
                <div class="menu-item" data-page="reporter">
                    <span class="icon">👤</span>
                    <span>报警人输入</span>
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
                <div class="camera-item" data-camera="1">
                    <span class="status-dot online"></span>
                    <span class="camera-icon">📹</span>
                    <span>摄像头 1</span>
                </div>
                <div class="camera-item" data-camera="2">
                    <span class="status-dot online"></span>
                    <span class="camera-icon">📹</span>
                    <span>摄像头 2</span>
                </div>
                <div class="camera-item" data-camera="3">
                    <span class="status-dot offline"></span>
                    <span class="camera-icon">📹</span>
                    <span>摄像头 3</span>
                </div>
                <div class="camera-item" data-camera="4">
                    <span class="status-dot online"></span>
                    <span class="camera-icon">📹</span>
                    <span>摄像头 4</span>
                </div>
            </div>
            <div class="search-box">
                <input type="text" placeholder="搜索事件..." id="searchInput">
            </div>
        </aside>

        <!-- 主内容区 -->
        <main class="main-content">
            <!-- 左侧区域 (视频 + 统计卡片) -->
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
                    <span>ℹ️ 自动刷新 | 当前: <span id="currentTime">{{ current_time.strftime('%Y-%m-%d %H:%M:%S') }}</span></span>
                    <span id="connectionStatus" style="color: var(--success-green);">● 已连接</span>
                </div>

                <div class="video-section">
                    <div class="video-container">
                        <img src="{{ url_for('video_feed') }}" alt="监控视频流" id="videoFeed">
                        <div class="video-overlay">
                            <div class="overlay-top">
                                <span>📹 Camera 01</span>
                                <span>1920x1080 @ 30FPS</span>
                            </div>
                            <div class="overlay-bottom">
                                <div><span class="ai-badge">🤖 AI Detection Active</span></div>
                                <span id="videoTime">{{ current_time.strftime('%Y-%m-%d %H:%M:%S') }}</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="alertCount">{{ alert_count }}</div>
                        <div class="stat-label">当前生成总量</div>
                        <div class="stat-change warning">↑ {{ percentage }}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="zoneTime">{{ zone_time }}s</div>
                        <div class="stat-label">异常个数</div>
                        <div class="stat-change info">今日事件</div>
                    </div>
                </div>
            </div>

            <!-- 右侧区域：异常片段 + 底部统计 -->
            <div class="right-panel">
                <div class="right-panel-header">
                    <h3>🚨 异常片段</h3>
                    <button class="btn" id="refreshEventsBtn" style="padding: 4px 8px;">🔄</button>
                </div>
                <!-- 底部异常次数统计卡片 -->
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

                <!-- 可滚动的事件列表 -->
                <div class="anomaly-list-container">
                    <div class="anomaly-list" id="anomalyList">
                        {% for event in events %}
                        <div class="anomaly-card priority-{{ event.priority }}" data-event='{{ event | tojson }}'>
                            <div class="thumbnail">
                                <div class="thumbnail-icon">
                                    {% if event.type == 'person' %}👤
                                    {% elif event.type == 'vehicle' %}🚗
                                    {% elif event.type == 'bicycle' %}🚲
                                    {% elif event.type == 'motion' %}🎬
                                    {% else %}🎥{% endif %}
                                </div>
                            </div>
                            <div class="anomaly-info">
                                <div class="anomaly-title">{{ event.type }}</div>
                                <div class="anomaly-meta">
                                    <span>{{ event.camera }}</span>
                                    <span class="confidence">{{ event.confidence }}%</span>
                                </div>
                                <div class="anomaly-time">{{ event.time }}</div>
                            </div>
                        </div>
                        {% else %}
                        <div style="text-align: center; color: var(--text-gray); padding: 20px;">暂无异常事件</div>
                        {% endfor %}
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- 弹窗用于模拟播放 -->
    <div id="eventModal" class="modal">
        <div class="modal-content">
            <h4 id="modalTitle">异常事件详情</h4>
            <p id="modalDetail">点击播放视频片段（模拟）</p>
            <button id="closeModal">关闭</button>
        </div>
    </div>

    <script>
        // 全局变量
        let refreshInterval, eventsInterval;
        let allEvents = [];   // 存储所有事件原始数据（含完整时间）
        let currentDateRange = 'today';

        // 辅助函数：格式化日期（YYYY-MM-DD）
        function formatDate(date) {
            return date.toISOString().slice(0,10);
        }

        // 判断事件是否属于某个范围
        function isEventInRange(eventTimeStr, range) {
            // eventTimeStr 格式 "HH:MM:SS" 但后端返回的 event.time 只包含时间，没有日期。
            // 但是 generate_events 里使用了 datetime.now() - timedelta(minutes=minutes_ago) 并格式化为 "%H:%M:%S"
            // 这意味着 event.time 只有时间，丢失了日期信息。为了按天统计，我们需要修改 Python 代码？不，前端可以基于当前时间+相对分钟推算。
            // 但是为了简单，我们可以让后端返回完整时间戳。这里由于无法修改后端，我们采用前端模拟增强：在刷新事件时，重新构造带日期的事件对象。
            // 我们将在 refreshAnomalyList 中重新构建 events 时，增加一个 simulatedDate 字段，基于当前时间减去随机分钟（或使用原后端的相对时间？后端返回的 time 字段已经失去了日期）。
            // 为了演示按天统计功能，我们将在前端模拟：给每个事件一个随机的今日或昨日日期。由于演示目的，可以接受。
            // 更优雅：后端修改 API 返回 datetime 字符串。但无法改动后端的情况下，我们模拟近期事件。
            // 实际上后端 generate_events 中 minutes_ago 范围 1-60，所以事件都在过去1小时内。因此所有事件都属于“今日”。
            // 为了展示昨日/本周效果，我们在前端模拟一些历史事件。为了方便，我们将在刷新时给每个事件加上一个基于 random 的日期偏移（仅用于演示）。
            // 但更好的方式是：让实际部署时后端返回完整时间。这里我们在前端模拟一个日期字段。
            // 由于时间关系，我们通过一个全局变量来判定：如果事件有原始时间字符串，就沿用；否则用当前时间减去几分钟。
            // 为了演示功能，我们将所有事件都视为今日发生（因为 minutes_ago <= 60），所以今日统计正常，昨日为0。用户可通过切换看到效果。
            // 实际生产环境后端应返回完整 ISO 时间。
            const now = new Date();
            // 假设每个事件的时间是今天的时间点（时分秒来自 event.time）
            const [hours, minutes, seconds] = eventTimeStr.split(':').map(Number);
            const eventDate = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hours, minutes, seconds);

            const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const yesterdayStart = new Date(todayStart);
            yesterdayStart.setDate(yesterdayStart.getDate() - 1);
            const weekStart = new Date(todayStart);
            weekStart.setDate(weekStart.getDate() - now.getDay());

            if (range === 'today') {
                return eventDate >= todayStart;
            } else if (range === 'yesterday') {
                return eventDate >= yesterdayStart && eventDate < todayStart;
            } else if (range === 'week') {
                return eventDate >= weekStart;
            }
            return false;
        }

        // 更新底部异常次数统计
        function updateAnomalyCount() {
            let filtered = allEvents;
            if (currentDateRange === 'today') {
                filtered = allEvents.filter(ev => isEventInRange(ev.time, 'today'));
            } else if (currentDateRange === 'yesterday') {
                filtered = allEvents.filter(ev => isEventInRange(ev.time, 'yesterday'));
            } else if (currentDateRange === 'week') {
                filtered = allEvents.filter(ev => isEventInRange(ev.time, 'week'));
            }
            document.getElementById('anomalyCount').innerText = filtered.length;
        }

        // 刷新异常列表（从后端获取事件，并保存到 allEvents）
        function refreshAnomalyList() {
            fetch('/api/events')
                .then(res => res.json())
                .then(data => {
                    allEvents = data;  // 存储原始事件（time 字段仅时间，日期为今天）
                    // 渲染列表
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
                        card.className = `anomaly-card priority-${event.priority}`;
                        // 构建缩略图HTML
                        let thumbnailHtml = '';
                        if (event.thumbnail_url) {
                            thumbnailHtml = `
                                <div class="thumbnail">
                                    <img src="${event.thumbnail_url}" alt="异常缩略图" style="width: 100%; height: 100%; object-fit: cover;">
                                    <div class="thumbnail-overlay">
                                        <span class="play-icon">▶</span>
                                    </div>
                                </div>
                            `;
                        } else {
                            // 降级方案：显示图标
                            let icon = '🎥';
                            if (event.type === 'person') icon = '👤';
                            else if (event.type === 'vehicle') icon = '🚗';
                            else if (event.type === 'bicycle') icon = '🚲';
                            else if (event.type === 'motion') icon = '🎬';
                            thumbnailHtml = `
                                <div class="thumbnail">
                                    <div class="thumbnail-icon">${icon}</div>
                                </div>
                            `;
                        }
                        card.innerHTML = `
                            <div class="thumbnail">
                                <div class="thumbnail-icon">${icon}</div>
                            </div>
                            <div class="anomaly-info">
                                <div class="anomaly-title">${event.type}</div>
                                <div class="anomaly-meta">
                                    <span>${event.camera}</span>
                                    <span class="confidence">${event.confidence}%</span>
                                </div>
                                <div class="anomaly-time">${event.time}</div>
                            </div>
                        `;
                        // 存储事件数据到 dataset，用于点击
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

        // 点击事件弹出模态框（播放模拟）
        function showEventModal(event) {
            const modal = document.getElementById('eventModal');
            const modalTitle = document.getElementById('modalTitle');
            const modalDetail = document.getElementById('modalDetail');
            modalTitle.innerText = `${event.type} 异常片段`;
            modalDetail.innerHTML = `摄像机: ${event.camera}<br>时间: ${event.time}<br>置信度: ${event.confidence}%<br><br>🎬 点击播放按钮查看片段（模拟）<br><button id="playMockBtn" style="margin-top:10px;">播放片段</button>`;
            modal.style.display = 'flex';
            const playBtn = document.getElementById('playMockBtn');
            if (playBtn) {
                playBtn.onclick = () => {
                    alert(`正在播放片段：${event.type}，来自 ${event.camera} 于 ${event.time}\n实际可接入视频回放服务。`);
                };
            }
        }

        // 关闭模态框
        function closeModal() {
            document.getElementById('eventModal').style.display = 'none';
        }

        // 刷新左侧统计卡片
        function updateStats() {
            fetch('/refresh_data')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('alertCount').innerText = data.count;
                    document.getElementById('zoneTime').innerText = data.in_zone_time + 's';
                    if (data.targets) {
                        const total = data.targets.person + data.targets.vehicle + data.targets.bicycle;
                        document.getElementById('totalTargets').innerText = total;
                        document.getElementById('targetStats').innerHTML = `${data.targets.person} 人, ${data.targets.vehicle} 车, ${data.targets.bicycle} 自行车`;
                    }
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

        function switchCamera(cameraId) {
            const videoImg = document.getElementById('videoFeed');
            if (videoImg) {
                videoImg.src = videoImg.src.split('?')[0] + `?camera=${cameraId}&t=${Date.now()}`;
                document.querySelector('.overlay-top span:first-child').innerText = `📹 Camera ${cameraId.toString().padStart(2, '0')}`;
                showNotification(`切换到 Camera ${cameraId}`, 'success');
            }
        }

        function filterAnomalies() {
            const keyword = document.getElementById('searchInput').value.toLowerCase();
            const cards = document.querySelectorAll('.anomaly-card');
            cards.forEach(card => {
                const text = card.textContent.toLowerCase();
                card.style.display = text.includes(keyword) ? 'flex' : 'none';
            });
        }

        // 连接检测
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

        function updateTime() {
            const now = new Date();
            const timeStr = now.toISOString().slice(0, 19).replace('T', ' ');
            document.getElementById('currentTime').innerText = timeStr;
            document.getElementById('videoTime').innerText = timeStr;
        }

        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 16px;
                background-color: ${type === 'success' ? '#2ed573' : type === 'error' ? '#ff4757' : '#00d4ff'};
                color: white;
                border-radius: 6px;
                font-size: 13px;
                z-index: 9999;
                animation: slideIn 0.3s ease;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;
            notification.textContent = message;
            document.body.appendChild(notification);
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 2500);
            }, 2500);
        }

        // 初始化
        function init() {
            document.getElementById('screenshotBtn').addEventListener('click', takeScreenshot);
            document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
            document.getElementById('refreshDataBtn').addEventListener('click', () => {
                updateStats();
                showNotification('数据已刷新', 'success');
            });
            document.getElementById('refreshEventsBtn').addEventListener('click', refreshAnomalyList);
            document.getElementById('searchInput').addEventListener('input', filterAnomalies);
            document.getElementById('closeModal').addEventListener('click', closeModal);
            // 日期切换
            document.querySelectorAll('.date-tab').forEach(tab => {
                tab.addEventListener('click', function() {
                    document.querySelectorAll('.date-tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    currentDateRange = this.dataset.range;
                    updateAnomalyCount();
                });
            });

            document.querySelectorAll('.camera-item').forEach(item => {
                item.addEventListener('click', function() {
                    if (this.querySelector('.status-dot').classList.contains('online')) {
                        switchCamera(this.dataset.camera);
                    } else {
                        showNotification('摄像头离线', 'error');
                    }
                });
            });

            // 菜单占位
            document.querySelectorAll('.menu-item').forEach(item => {
                item.addEventListener('click', function() {
                    if (!this.classList.contains('active')) {
                        showNotification('功能开发中', 'info');
                    }
                });
            });

            // 定时器
            refreshAnomalyList();
            updateStats();
            refreshInterval = setInterval(updateStats, 5000);
            eventsInterval = setInterval(refreshAnomalyList, 10000);
            setInterval(updateTime, 1000);
            setInterval(checkVideoConnection, 30000);

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

            console.log('SOP天眼监控系统已启动（优化布局+异常片段可点击+按天统计）');
        }

        document.addEventListener('DOMContentLoaded', init);

        // 动画样式（动态添加）
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
