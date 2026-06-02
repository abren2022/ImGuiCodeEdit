from flask import Flask, render_template, Response, jsonify, url_for, request
import random
import os
import json
from datetime import datetime, timedelta
import cv2
import urllib.parse
import time
import threading
from AsseBehaviorMonitoring import Config
#,SOPMonitor,YOLODetector
# from AsseBehaviorMonitoring import alert_manager
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
        self.cameras = []          # 存储摄像头信息列表
        self.camera_instances = {}        # camera_id -> VideoCamera 实例
        self.active_camera_id = None
        self.lock = threading.RLock()
        self.load_from_file()

    def load_from_file(self):
        """从 JSON 文件加载摄像头配置（强制 UTF-8）"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.cameras = json.load(f)
                print(f"加载了 {len(self.cameras)} 个摄像头")
            except UnicodeDecodeError:
                # 如果文件不是 UTF-8 编码，尝试用 GBK 读取并重新保存为 UTF-8
                print("文件编码不是 UTF-8，尝试用 GBK 读取...")
                with open(self.config_file, 'r', encoding='gbk') as f:
                    self.cameras = json.load(f)
                # 重新保存为 UTF-8
                self._save_to_file()
                print("已将配置文件转换为 UTF-8 编码")
        else:
            print(f"配置文件 {self.config_file} 不存在，将创建默认摄像头")
            default_cam = {
                'id': 1,
                'name': '默认摄像头',
                'rtsp': config.source_url,
                'status': 'online'
            }
            self.cameras = [default_cam]
            self._save_to_file()


    def _save_to_file(self):
        """保存摄像头列表到 JSON 文件（UTF-8 编码，保留中文）"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.cameras, f, indent=2, ensure_ascii=False)
        print(f"已保存 {len(self.cameras)} 个摄像头配置到 {self.config_file}")

    def get_list(self):
        """返回前端需要的摄像机列表（不含敏感密码）"""
        return [{
            'id': c['id'],
            'name': c['name'],
            'status': c.get('status', 'online')
        } for c in self.cameras]

    def add_camera(self, name, rtsp_url):
        """新增摄像头，自动分配 ID"""
        new_id = max([c['id'] for c in self.cameras], default=0) + 1
        new_cam = {
            'id': new_id,
            'name': name,
            'rtsp': rtsp_url,
            'status': 'online'
        }
        self.cameras.append(new_cam)
        self._save_to_file()
        # 创建对应的 VideoCamera 实例（但不自动切换）
        self.camera_instances[new_id] = VideoCamera(rtsp_url)
        return new_id

    def remove_camera(self, camera_id):
        """删除摄像头（如果正在使用则先切换到其他）"""
        # 找到并删除
        to_remove = None
        for c in self.cameras:
            if c['id'] == camera_id:
                to_remove = c
                break
        if not to_remove:
            return False
        self.cameras.remove(to_remove)
        self._save_to_file()
        # 释放实例
        if camera_id in self.camera_instances:
            del self.camera_instances[camera_id]
        # 如果删除的是当前激活的摄像头，尝试切换到第一个可用的
        if self.active_camera_id == camera_id:
            if self.cameras:
                self.switch_to(self.cameras[0]['id'])
            else:
                self.active_camera_id = None
        return True

    def switch_to(self, camera_id):
        with self.lock:  # 假设 self.lock 已在 __init__ 中定义
            # 如果已经是当前激活的摄像头，无需切换
            if self.active_camera_id == camera_id:
                return self.camera_instances.get(camera_id)

            # 释放当前激活的摄像头资源
            if self.active_camera_id is not None and self.active_camera_id in self.camera_instances:
                old_cam = self.camera_instances[self.active_camera_id]
                old_cam.release()  # 需要 VideoCamera 实现 release 方法
                # 可选择删除实例，或者保留但连接已关闭
                # 为了避免下次切换回来重新创建，可以保留但标记为未初始化，但简单期间可以删除
                del self.camera_instances[self.active_camera_id]
                time.sleep(1)

            # 获取或创建新摄像头实例
            if camera_id not in self.camera_instances:
                cam_info = next(c for c in self.cameras if c['id'] == camera_id)
                self.camera_instances[camera_id] = VideoCamera(cam_info['rtsp'])

            self.active_camera_id = camera_id
            return self.camera_instances[camera_id]

    def get_active_frame(self):
        """获取当前激活摄像头的帧"""
        if self.active_camera_id and self.active_camera_id in self.camera_instances:
            cam = self.camera_instances[self.active_camera_id]
            if cam.cap is None or not cam.cap.isOpened():
                # 尝试重新连接
                cam.release()
                cam.initialize_camera()
            return cam.get_frame()
        return None

    def reconfigure_active(self, new_rtsp_url, new_name=None):
        """
        重新配置当前激活的摄像头（更新 RTSP 和名称），同时更新配置文件
        """
        if self.active_camera_id is None:
            return False, "没有激活的摄像头"
        # 找到对应配置项
        for cam in self.cameras:
            if cam['id'] == self.active_camera_id:
                old_rtsp = cam['rtsp']
                cam['rtsp'] = new_rtsp_url
                if new_name:
                    cam['name'] = new_name
                self._save_to_file()
                # 重新创建实例
                if self.active_camera_id in self.camera_instances:
                    self.camera_instances[self.active_camera_id].__del__()  # 释放旧资源
                self.camera_instances[self.active_camera_id] = VideoCamera(new_rtsp_url)
                return True, "更新成功"
        return False, "未找到摄像头"

class VideoCamera:
    def __init__(self, camera_id):
        self.video_source = camera_id
        self.cap = None
        self.is_simulated = False
        self.lock = threading.RLock()
        self.initialize_camera()

    def initialize_camera(self):
        """初始化摄像头"""
        with self.lock:
            # 释放原有资源
            if self.cap:
                self.cap.release()
                self.cap = None
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

    def reconfigure(self, new_rtsp_url):
        """
        动态切换RTSP流
        :param new_rtsp_url: 完整的RTSP URL
        :return: (success, message)
        """
        with self.lock:
            # 保存原有配置以备回滚
            old_source = self.video_source
            old_cap = self.cap

            # 尝试连接新URL
            try:
                test_cap = cv2.VideoCapture(new_rtsp_url, cv2.CAP_FFMPEG)
                test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not test_cap.isOpened():
                    return False, "无法连接到RTSP流，请检查参数"

                ret, test_frame = test_cap.read()
                if not ret or test_frame is None:
                    test_cap.release()
                    return False, "RTSP流无法获取视频帧"

                # 测试通过，正式切换
                if self.cap:
                    self.cap.release()
                self.video_source = new_rtsp_url
                self.cap = test_cap
                self.is_simulated = False
                print(f"成功切换到新摄像头: {new_rtsp_url}")
                return True, "摄像头切换成功"

            except Exception as e:
                # 切换失败，回滚
                print(f"切换失败: {e}")
                if old_cap:
                    self.cap = old_cap
                    self.video_source = old_source
                else:
                    self.cap = None
                    self.is_simulated = True
                return False, f"切换失败: {str(e)}"

    def get_frame(self):
        """获取视频帧"""
        if not self.is_simulated and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # detections = detector.detect_and_track(frame)
                # current_time = time.time()
                # monitor.update_all_workpieces(detections,current_time)
                # return monitor.draw_status(frame, detections)
                return  frame

        return None


    def release(self):
        with self.lock:
            if self.cap:
                # 先释放资源
                self.cap.release()
                # 显式删除对象引用，帮助垃圾回收
                self.cap = None
                # 等待一小段时间，让 FFmpeg 完全清理
                time.sleep(1)

def gen_frames():
    """生成视频流"""
    while True:
        try:
            #frame = camera.get_frame()
            frame = camera_manager.get_active_frame()
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
    # recent_alerts = alert_manager.get_recent_alerts(limit=10)
    #
    # for alert in recent_alerts:
    #     # 2. 格式化数据以匹配前端需要的格式
    #     events.append({
    #         'type': alert.alert_type,   # 例如: 'timeout', 'missing_step'
    #         'time': datetime.fromtimestamp(alert.timestamp).strftime("%H:%M:%S"),
    #         'camera': 'Camera 01',  # 如果有多个摄像头，这里需要动态获取
    #         'description': alert.description,  # 额外字段，前端可选显示
    #         'video_url':alert.video_clip_path if alert.video_clip_path else ''  # 如果有视频片段路径
    #     })

    # 如果历史记录为空，返回空列表或默认提示
    if not events:
        # 可选：返回一个默认的“无报警”状态，或者保持为空
        pass

    return events

# 初始化全局摄像头管理器（需在 detector/monitor 等初始化之后）
camera_manager = CameraManager(CAMERAS_CONFIG_FILE)

# 修改原有的全局 camera 变量为使用 manager 的当前实例
def get_current_camera():
    return camera_manager.camera_instances.get(camera_manager.active_camera_id)
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
        'camera_status': 'simulated' if camera_manager.camera_instances[camera_manager.active_camera_id].is_simulated else 'connected',
        'timestamp': datetime.now().isoformat()
    })


# 修改截图功能，返回更详细的信息
@app.route('/screenshot')
def screenshot():
    """截图功能"""
    try:
        frame = camera_manager.camera_instances[camera_manager.active_camera_id].get_frame()
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
        required_fields = ['ip', 'port', 'username', 'password', 'channel','point_name']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'message': f'缺少必要字段: {field}'}), 400

        ip = data['ip']
        port = int(data['port'])
        username = data['username']
        password = data['password']
        channel = data['channel']
        camera_name = data['point_name']

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
    # detector = YOLODetector(config)
    # monitor = SOPMonitor(config)

    print(f"🌐 访问地址: http://localhost:5000")
    print(f"📹 视频流地址: http://localhost:5000/video_feed")
    print("=" * 50)

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)
