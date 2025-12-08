import panel as pn
import param
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import os
import zipfile
import tempfile
import shutil
import sys
import subprocess
import re
import queue
import time
import threading
# 启用必要的扩展
pn.extension('plotly')


class TrainingPage(param.Parameterized):
    # 模型参数
    model = param.Selector(default='efficientAD',
                           objects=['efficientAD', 'fastflow', 'padim', 'patchcore', 'reversedistillation', 'GLASS'])
    script_path = param.String(default="")

    # 训练参数
    max_epochs = param.Integer(default=100, bounds=(1, 1000))
    batch_size = param.Selector(default=32, objects=[16, 32, 64, 128, 256])
    learning_rate = param.Number(default=0.001, bounds=(0.0001, 1.0))
    optimizer = param.Selector(default='Adam', objects=['Adam', 'SGD', 'RMSprop'])
    weight_decay = param.Number(default=0.0001, bounds=(0.0, 0.01))
    val_ratio = param.Number(default=0.2, bounds=(0.1, 0.5))

    # 路径参数
    data_dir = param.String(default="")
    output_dir = param.String(default="")

    # 训练状态
    is_training = param.Boolean(default=False)
    training_logs = param.List(default=[])
    training_data = param.Dict(default={'loss': [], 'f1': [], 'auc': []})
    current_epoch = param.Integer(default=0)

    # 上传和下载相关
    upload_filename = param.String(default="")
    download_filename = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self.process = None
        self.epoch = -1
        self.temp_dir = tempfile.mkdtemp()
        self.uploaded_file = None
        self.download_file = None
        self._log_counter = 0  # 添加计数器


    def __del__(self):
        # 清理临时目录
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def remove_ansi_escape_sequences(self, text: str):
        try:
            ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
            return ansi_escape.sub('', text)
        except UnicodeDecodeError:
            # 如果遇到编码问题，尝试其他编码方式
            return text.encode('utf-8', errors='ignore').decode('utf-8')

    def clean_output(self, text: str):
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    @param.depends('is_training', watch=True)
    def update_button_state(self):
        if hasattr(self, 'start_button'):
            if self.is_training:
                self.start_button.name = "训练中..."
                self.start_button.button_type = 'danger'
            else:
                self.start_button.name = "开始训练"
                self.start_button.button_type = 'primary'

    def add_log(self, message):
        import datetime
        output = self.remove_ansi_escape_sequences(message)
        output = self.clean_output(output)
        loss_pattern = re.compile(r"train_loss_epoch=(\d+\.\d+)")
        auc_pattern = re.compile(r"image_AUROC=(\d+\.\d+)")
        f1_pattern = re.compile(r"image_F1Score=(\d+\.\d+)")
        for line in output.splitlines():
            if "Epoch" in line:
                match = re.search(r'Epoch (\d+)', line)
                self.current_epoch = self.epoch + 1
                loss_match = loss_pattern.search(line)
                f1_match = f1_pattern.search(line)
                auc_match = auc_pattern.search(line)
                loss = float(loss_match.group(1)) if loss_match else 15.0
                f1 = float(f1_match.group(1)) if f1_match else 0.0
                auc = float(auc_match.group(1)) if auc_match else 0.0
                # 更新训练数据
                if self.epoch < len(self.training_data['loss']):
                    self.training_data['loss'][self.epoch] = loss
                    self.training_data['f1'][self.epoch] = f1
                    self.training_data['auc'][self.epoch] = auc
                else:
                    self.training_data['loss'].append(loss)
                    self.training_data['f1'].append(f1)
                    self.training_data['auc'].append(auc)

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_logs.append(log_entry)
        if len(self.training_logs) > 100:
            self.training_logs.pop(0)
        self._log_counter += 1
        # 每3条日志或包含关键信息时才触发更新
        if self._log_counter % 3 == 0 or any(keyword in message for keyword in ["Epoch", "训练完成", "训练失败"]):
            self.param.trigger('training_logs')


    def select_data_dir(self, event):
        self.data_dir = "D:/datasets/weding"
        self.add_log(f"已选择数据目录: {self.data_dir}")

    def select_output_dir(self, event):
        self.output_dir = "./train_results"
        self.add_log(f"已选择输出目录: {self.output_dir}")

    def handle_upload(self, event):
        """处理文件上传"""
        if not event.new:
            return

        # 获取上传的文件
        file_data = event.new
        filename = event.obj.filename

        if filename.endswith('.zip'):
            self.uploaded_file = file_data
            self.upload_filename = filename
            self.add_log(f"已上传数据文件: {filename}")

            # 自动解压到临时目录
            try:
                extract_path = os.path.join(self.temp_dir, "uploaded_data")
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path)
                os.makedirs(extract_path)

                # 保存并解压ZIP文件
                zip_path = os.path.join(self.temp_dir, filename)
                with open(zip_path, 'wb') as f:
                    f.write(file_data)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                self.data_dir = extract_path
                self.add_log(f"数据已解压到: {extract_path}")
                self.add_log("请检查数据结构是否符合要求")

            except Exception as e:
                self.add_log(f"解压文件时出错: {str(e)}")
        else:
            self.add_log("请上传ZIP格式的数据文件")

    def create_download_file(self):
        """创建模拟的模型下载文件"""
        try:
            # 模拟创建模型文件 - 在实际应用中，这里应该是真实的模型文件
            model_content = f"""
            # AI Model File
            Model: {self.model}
            Training Epochs: {self.max_epochs}
            Batch Size: {self.batch_size}
            Learning Rate: {self.learning_rate}
            Final Loss: {self.training_data['loss'][-1]:.4f if self.training_data['loss'] else 0}
            Final F1 Score: {self.training_data['f1'][-1]:.4f if self.training_data['f1'] else 0}
            Final AUROC: {self.training_data['auc'][-1]:.4f if self.training_data['auc'] else 0}
            """.encode('utf-8')

            self.download_file = model_content
            self.download_filename = f"{self.model}_model_epoch{self.max_epochs}.pth"

            # 在实际应用中，这里应该返回真实的模型文件路径
            model_path = os.path.join(self.temp_dir, self.download_filename)
            with open(model_path, 'wb') as f:
                f.write(model_content)

            return model_path
        except Exception as e:
            self.add_log(f"创建下载文件时出错: {str(e)}")
            return None

    def download_model(self, event):
        """处理模型下载"""
        if not self.training_data['loss']:
            self.add_log("错误: 没有训练完成的模型可供下载")
            return

        model_path = self.create_download_file()
        if model_path and os.path.exists(model_path):
            self.add_log(f"模型文件已准备就绪: {self.download_filename}")

            # 更新下载按钮的文件和文件名
            if hasattr(self, 'download_button'):
                self.download_button.file = model_path
                self.download_button.filename = self.download_filename
        else:
            self.add_log("错误: 无法创建模型文件")

    def start_training(self, event):
        if self.is_training:
            self.add_log("训练正在进行中，请等待完成")
            return
        if self.model == 'efficientAD':
            self.script_path = "./train_anomal_image/train_efficientAD.py"
        elif self.model == 'fastflow':
            self.script_path = "./train_anomal_image/train_fastflow.py"
        elif self.model == 'padim':
            self.script_path = "./train_anomal_image/train_padim.py"
        elif self.model == 'patchcore':
            self.script_path = "./train_anomal_image/train_patchcore.py"
        elif self.model == 'reversedistillation':
            self.script_path = "./train_anomal_image/train_reversedistillation.py"
        elif self.model == 'GLASS':
            self.script_path = "./train_anomal_image/train_GLASS.py"
        if not all([self.script_path, self.data_dir, self.output_dir]):
            self.add_log("错误: 请选择训练脚本、数据目录和输出目录")
            return

        # 检查数据目录是否存在
        if not os.path.exists(self.data_dir):
            self.add_log("错误: 数据目录不存在，请先上传数据")
            return

        self.is_training = True
        self.current_epoch = 0
        self.training_data = {'loss': [], 'f1': [], 'auc': []}

        thread = threading.Thread(target=self.run_training)
        thread.daemon = True
        thread.start()

    def run_training(self):
        try:
            command = [
                sys.executable, self.script_path,
                "--dataset_root", self.data_dir,
                "--max_epochs", str(self.max_epochs),
                "--output_dir", self.output_dir,
            ]

            self.add_log(f"开始训练，命令: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                encoding='utf-8',
                bufsize=0,  # 无缓冲
                env=dict(os.environ, PYTHONUNBUFFERED='1')  # 禁用Python缓冲
            )
            while True:
                line = process.stdout.readline()
                if line:
                    self.add_log(line.strip())
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            process.wait()

            if process.returncode == 0:
                self.add_log("训练完成!")
                self.create_download_file()
            else:
                self.add_log(f"训练失败，退出码: {process.returncode}")

        except Exception as e:
            self.add_log(f"训练出错: {str(e)}")
        finally:
            self.is_training = False

    def stop_training(self, event):
        self.is_training = False
        self.add_log("正在停止训练...")

    @param.depends('training_data')
    def plot_training_curves(self):
        if not self.training_data['loss']:
            return pn.pane.Markdown("## 训练曲线\n\n训练开始后将显示图表")

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Loss曲线', 'F1 Score', 'AUROC'),
            vertical_spacing=0.06,
            shared_xaxes=True
        )

        max_epochs = list(range(1, len(self.training_data['loss']) + 1))

        fig.add_trace(
            go.Scatter(
                x=max_epochs, y=self.training_data['loss'],
                mode='lines+markers',
                name='Loss',
                line=dict(color='#C73E1D', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=max_epochs, y=self.training_data['f1'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='#3DA35D', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=max_epochs, y=self.training_data['auc'],
                mode='lines+markers',
                name='AUROC',
                line=dict(color='#2E86AB', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )

        fig.update_xaxes(title_text="训练轮次", row=3, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="F1 Score", row=2, col=1)
        fig.update_yaxes(title_text="AUROC", row=3, col=1)

        return pn.pane.Plotly(fig, height=400)

    @param.depends('training_logs')
    def log_train_view(self):
        logs_text = "\n".join(self.training_logs[-15:])
        if not logs_text:
            return pn.pane.Markdown("暂无日志")

        # 使用简单的 Markdown 代码块格式
        return pn.pane.Markdown(f"```\n{logs_text}\n```")

    @param.depends('training_data', 'current_epoch')
    def training_stats(self):
        if not self.training_data['loss']:
            return pn.pane.Markdown("## 等待训练开始...")

        current_loss = self.training_data['loss'][-1] if self.training_data['loss'] else 0
        current_f1 = self.training_data['f1'][-1] if self.training_data['f1'] else 0
        current_auc = self.training_data['auc'][-1] if self.training_data['auc'] else 0

        # 使用简单的 HTML 表格来显示统计信息
        stats_html = f"""
        <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
            <tr>
                <td style="text-align: center; padding: 8px; border: 1px solid #ddd;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #2E86AB;">{current_loss:.4f}</div>
                    <div style="font-size: 0.8em; color: #666;">Loss</div>
                </td>
                <td style="text-align: center; padding: 8px; border: 1px solid #ddd;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #2E86AB;">{current_f1:.4f}</div>
                    <div style="font-size: 0.8em; color: #666;">F1 Score</div>
                </td>
                <td style="text-align: center; padding: 8px; border: 1px solid #ddd;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #2E86AB;">{current_auc:.4f}</div>
                    <div style="font-size: 0.8em; color: #666;">AUROC</div>
                </td>
                <td style="text-align: center; padding: 8px; border: 1px solid #ddd;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #2E86AB;">{self.current_epoch}/{self.max_epochs}</div>
                    <div style="font-size: 0.8em; color: #666;">Epoch</div>
                </td>
            </tr>
        </table>
        """

        progress = (self.current_epoch / self.max_epochs) * 100 if self.max_epochs > 0 else 0
        progress_html = f"""
        <div style="background: #f0f0f0; border-radius: 10px; padding: 5px; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, #2E86AB, #A23B72); height: 8px; border-radius: 5px; width: {progress}%;"></div>
        </div>
        <div style="text-align: center; font-size: 0.9em; color: #666;">
            进度: {self.current_epoch}/{self.max_epochs} ({progress:.1f}%)
        </div>
        """

        return pn.Column(
            pn.pane.Markdown("## 训练统计"),
            pn.pane.HTML(stats_html),
            pn.pane.HTML(progress_html)
        )

    def view(self):

        data_button = pn.widgets.Button(name="浏览", button_type="default", height=31, margin=(25, 0, 0, 0))
        data_button.on_click(self.select_data_dir)

        output_button = pn.widgets.Button(name="浏览", button_type="default", height=31, margin=(25, 0, 0, 0))
        output_button.on_click(self.select_output_dir)

        self.start_button = pn.widgets.Button(name="开始训练", button_type="primary")
        self.start_button.on_click(self.start_training)

        stop_button = pn.widgets.Button(name="停止训练", button_type="warning")
        stop_button.on_click(self.stop_training)

        # 文件上传组件
        file_upload = pn.widgets.FileInput(
            name='上传训练数据 (ZIP格式)',
            accept='.zip',
            multiple=False
        )
        file_upload.param.watch(self.handle_upload, 'value')

        # 模型下载组件
        self.download_button = pn.widgets.FileDownload(
            filename='model.pth',
            button_type='success',
            auto=False
        )
        download_btn = pn.widgets.Button(name="下载训练模型", button_type="success")
        download_btn.on_click(self.download_model)

        # 数据上传区域
        data_upload_card = pn.Card(
            pn.Column(
                file_upload,
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.data_dir,name="数据目录", width=100),
                    data_button
                ),
            ),
            styles={'padding': '10px'},
            title="本地zip数据上传"
        )

        # 模型配置区域
        model_config = pn.Card(
            pn.Column(
                pn.Row(
                    pn.widgets.Select.from_param(self.param.model, name="训练模型", width=200)
                ),
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.output_dir, name="模型保存路径", width=250),
                    output_button
                )
            ),
            title="模型配置",
            styles={'padding': '10px'},
            min_height=180
        )

        # 训练参数区域
        training_params = pn.Card(
            pn.GridBox(
                pn.widgets.IntSlider.from_param(self.param.max_epochs, name="训练轮次"),
                pn.widgets.Select.from_param(self.param.batch_size, name="批大小"),
                pn.widgets.NumberInput.from_param(self.param.learning_rate, name="学习率"),
                pn.widgets.Select.from_param(self.param.optimizer, name="优化器"),
                pn.widgets.NumberInput.from_param(self.param.weight_decay, name="权重衰减"),
                pn.widgets.NumberInput.from_param(self.param.val_ratio, name="验证集比例"),
                ncols=3,
                sizing_mode="fixed",
                width=950,
                height=200
            ),
            title="训练参数",
            styles={'padding': '10px'},
            min_height=180
        )

        # 训练控制区域
        training_control = pn.Card(
            pn.Column(
                pn.Row(self.start_button, stop_button),
                pn.Row(download_btn, self.download_button),
                align = "center"
            ),
            title="训练控制",
            styles={'padding': '10px'},
            min_height=180
        )
        training_stats_card = pn.Card(
            self.training_stats,
            title="训练统计",
            styles={'padding': '10px'},
            min_height=180
        )
        # 训练曲线和日志区域
        training_visualization = pn.Card(
            pn.Row(
                pn.Column(
                    self.plot_training_curves,
                    width=698,
                    height=300
                ),
                pn.Column(
                    pn.pane.Markdown("## 训练日志"),
                    self.log_train_view,
                    width=396,
                    height=300
                )
            ),
            title="训练监控",
            styles={'padding': '10px'}
        )
        return pn.Column(
            pn.pane.Markdown("# 模型训练"),
            pn.Column(
                pn.Row(
                    pn.Column(
                        model_config,
                        width=380,
                        height=210
                    ),
                    pn.Column(
                        data_upload_card,
                        width=320,
                        height=210
                    ),
                    pn.Column(
                        training_control,
                        width=250,
                        height=210
                    ),
                    pn.Column(
                        training_stats_card,
                        width=300,
                        height=210
                    ),
                    sizing_mode="fixed",
                ),
                pn.Row(
                    pn.Column(
                        training_params,
                        width=1150,
                        height=230
                    )
                ),
            ),
            training_visualization,
            sizing_mode="stretch_width",
        )



class TestingPage(param.Parameterized):
    # 测试参数
    model_path = param.String(default="")
    config_path = param.String(default="")
    test_data_path = param.String(default="")
    test_script_path = param.String(default="")
    output_path = param.String(default="")

    # 测试状态
    is_testing = param.Boolean(default=False)
    test_logs = param.List(default=[])
    test_image = param.String(default="")

    # 上传相关
    upload_test_filename = param.String(default="")

    def __init__(self, **params):
        super().__init__(**params)
        self.process = None
        self.temp_dir = tempfile.mkdtemp()
        self.uploaded_test_file = None

    def __del__(self):
        # 清理临时目录
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @param.depends('is_testing', watch=True)
    def update_button_state(self):
        if hasattr(self, 'test_button'):
            if self.is_testing:
                self.test_button.name = "测试中..."
                self.test_button.button_type = 'danger'
            else:
                self.test_button.name = "开始测试"
                self.test_button.button_type = 'primary'

    def add_log(self, message):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.test_logs.append(log_entry)
        if len(self.test_logs) > 100:
            self.test_logs.pop(0)

    def select_model_file(self, event):
        self.model_path = "/path/to/model/file.pth"
        self.add_log(f"已选择模型文件: {self.model_path}")

    def select_config_file(self, event):
        self.config_path = "/path/to/config/file.json"
        self.add_log(f"已选择配置文件: {self.config_path}")

    def select_test_data_file(self, event):
        self.test_data_path = "/path/to/test/data"
        self.add_log(f"已选择测试数据: {self.test_data_path}")

    def select_test_data_folder(self, event):
        self.test_data_path = "/path/to/test/folder"
        self.add_log(f"已选择测试数据文件夹: {self.test_data_path}")

    def select_test_script(self, event):
        self.test_script_path = "/path/to/test/script.py"
        self.add_log(f"已选择测试脚本: {self.test_script_path}")

    def select_output_folder(self, event):
        self.output_path = "/path/to/output/folder"
        self.add_log(f"已选择输出文件夹: {self.output_path}")

    def handle_test_upload(self, event):
        """处理测试数据上传"""
        if not event.new:
            return

        # 获取上传的文件
        file_data = event.new
        filename = event.obj.filename

        if filename.endswith('.zip'):
            self.uploaded_test_file = file_data
            self.upload_test_filename = filename
            self.add_log(f"已上传测试数据文件: {filename}")

            # 自动解压到临时目录
            try:
                extract_path = os.path.join(self.temp_dir, "uploaded_test_data")
                if os.path.exists(extract_path):
                    shutil.rmtree(extract_path)
                os.makedirs(extract_path)

                # 保存并解压ZIP文件
                zip_path = os.path.join(self.temp_dir, filename)
                with open(zip_path, 'wb') as f:
                    f.write(file_data)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)

                self.test_data_path = extract_path
                self.add_log(f"测试数据已解压到: {extract_path}")

            except Exception as e:
                self.add_log(f"解压测试文件时出错: {str(e)}")
        else:
            self.add_log("请上传ZIP格式的测试数据文件")

    def run_test(self, event):
        if self.is_testing:
            self.add_log("测试正在进行中，请等待完成")
            return

        if not all([self.model_path, self.config_path, self.test_data_path,
                    self.test_script_path, self.output_path]):
            self.add_log("错误: 请填写所有必填项")
            return

        self.is_testing = True
        self.test_image = ""

        thread = threading.Thread(target=self.run_testing)
        thread.daemon = True
        thread.start()

    def run_testing(self):
        try:
            command = [
                "python", self.test_script_path,
                "--path_model", self.model_path,
                "--metadata", self.config_path,
                "--path_dataset", self.test_data_path,
                "--dir_result", self.output_path
            ]

            self.add_log(f"开始测试，命令: {' '.join(command)}")

            import time
            for i in range(5):
                if not self.is_testing:
                    break
                self.add_log(f"测试进度: {i + 1}/5")
                time.sleep(1)

            if self.is_testing:
                self.add_log("测试完成!")
                self.generate_sample_image()

        except Exception as e:
            self.add_log(f"测试出错: {str(e)}")
        finally:
            self.is_testing = False

    def generate_sample_image(self):
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        fig, ax = plt.subplots(figsize=(6, 4))

        cm = np.array([[45, 5], [3, 47]])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        classes = ['正常', '异常']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_title("混淆矩阵")
        ax.set_ylabel('真实标签')
        ax.set_xlabel('预测标签')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self.test_image = f"data:image/png;base64,{image_base64}"
        plt.close(fig)

    @param.depends('test_logs')
    def log_view(self):
        logs_text = "\n".join(self.test_logs[-15:])
        if not logs_text:
            return pn.pane.Markdown("暂无日志")

        return pn.pane.Markdown(f"```\n{logs_text}\n```")

    @param.depends('test_image')
    def image_view(self):
        if self.test_image:
            return pn.pane.HTML(f"""
                <div style="text-align: center;">
                    <h3>测试结果 - 混淆矩阵</h3>
                    <img src="{self.test_image}" style="max-width: 100%; border: 1px solid #e0e0e0; padding: 5px;">
                </div>
            """)
        else:
            return pn.pane.Markdown("""
                ## 测试结果

                测试完成后将显示结果图片
            """)

    def view(self):
        # 创建按钮
        model_button = pn.widgets.Button(name="浏览", button_type="default")
        model_button.on_click(self.select_model_file)

        config_button = pn.widgets.Button(name="浏览", button_type="default")
        config_button.on_click(self.select_config_file)

        file_button = pn.widgets.Button(name="选择文件", button_type="default")
        file_button.on_click(self.select_test_data_file)

        folder_button = pn.widgets.Button(name="选择文件夹", button_type="default")
        folder_button.on_click(self.select_test_data_folder)

        script_button = pn.widgets.Button(name="浏览", button_type="default")
        script_button.on_click(self.select_test_script)

        output_button = pn.widgets.Button(name="浏览", button_type="default")
        output_button.on_click(self.select_output_folder)

        self.test_button = pn.widgets.Button(name="开始测试", button_type="primary")
        self.test_button.on_click(self.run_test)

        # 测试数据上传组件
        test_upload = pn.widgets.FileInput(
            name='上传测试数据 (ZIP格式)',
            accept='.zip',
            multiple=False
        )
        test_upload.param.watch(self.handle_test_upload, 'value')

        # 测试数据上传区域
        test_upload_card = pn.Card(
            pn.Column(
                pn.pane.Markdown("### 本地上传测试数据"),
                pn.pane.Markdown("支持ZIP格式压缩包，上传后自动解压到临时目录"),
                test_upload,
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.test_data_path, name="测试数据路径", width=280),
                    pn.Row(file_button, folder_button)
                )
            ),
            title="测试数据上传"
        )

        # 测试配置区域
        test_config = pn.Card(
            pn.GridBox(
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.model_path, name="模型路径", width=280),
                    model_button
                ),
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.config_path, name="配置文件", width=280),
                    config_button
                ),
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.test_script_path, name="测试脚本", width=280),
                    script_button
                ),
                pn.Row(
                    pn.widgets.TextInput.from_param(self.param.output_path, name="输出路径", width=280),
                    output_button
                ),
                ncols=2
            ),
            title="测试配置"
        )

        # 结果显示区域
        test_results = pn.Card(
            pn.Row(
                pn.Column(
                    self.image_view,
                    width=450
                ),
                pn.Column(
                    pn.pane.Markdown("## 测试日志"),
                    self.log_view,
                    width=400
                )
            ),
            title="测试结果"
        )

        # 测试控制区域
        test_control = pn.Card(
            pn.Row(
                self.test_button,
                pn.pane.Markdown("确保所有配置项已填写完整")
            ),
            title="测试控制"
        )

        # 整体布局
        return pn.Column(
            pn.pane.Markdown("# 模型测试"),
            pn.Row(
                test_upload_card,
                test_config,
            ),
            test_control,
            test_results,
            sizing_mode="stretch_width"
        )


class AITrainingDashboard:
    def __init__(self):
        self.training_page = TrainingPage()
        self.testing_page = TestingPage()

    def view(self):
        tabs = pn.Tabs(
            ("模型训练", self.training_page.view()),
            ("模型测试", self.testing_page.view()),
            active=0
        )

        return tabs


# 创建并启动应用
dashboard = AITrainingDashboard()

# 设置模板
template = pn.template.FastListTemplate(
    title="AI模型训练与测试平台",
    sidebar=[],
    main=[dashboard.view()],
    accent="#2E86AB",
    header_background="#2E86AB"
)

# 启动应用
if __name__ == "__main__":
    template.show()
else:
    template.servable()
