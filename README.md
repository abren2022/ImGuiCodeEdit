import sys
import os
import re
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setupConnections()

    def initUI(self):
        self.setWindowTitle("模型训练测试系统")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon('./ai_icon.png'))  # 准备一个图标文件
        # 主容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 导航按钮
        self.nav_bar = self.createNavigationBar()
        main_layout.addWidget(self.nav_bar)

        # 页面堆栈
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # 初始化页面
        self.train_page = TrainingPage()
        self.test_page = TestingPage()
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.test_page)

        # 初始显示训练页面
        self.switchPage(0)

    def createNavigationBar(self):
        nav = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_train = QPushButton("模型训练")
        self.btn_test = QPushButton("模型测试")

        # 按钮样式
        style = """
            QPushButton {
                padding: 12px 24px;
                font-size: 14px;
                border: none;
                min-width: 120px;
            }
            .active {
                background-color: #4CAF50;
                color: white;
                border-radius: 4px;
            }
        """

        for btn in [self.btn_train, self.btn_test]:
            btn.setCheckable(True)
            btn.setStyleSheet(style)

        self.btn_train.setChecked(True)
        self.btn_train.setProperty("class", "active")

        layout.addWidget(self.btn_train)
        layout.addWidget(self.btn_test)
        layout.addStretch()
        nav.setLayout(layout)
        return nav

    def setupConnections(self):
        self.btn_train.clicked.connect(lambda: self.switchPage(0))
        self.btn_test.clicked.connect(lambda: self.switchPage(1))

    def switchPage(self, index):
        self.stack.setCurrentIndex(index)
        # 更新按钮状态
        self.btn_train.setProperty("class", "active" if index == 0 else "")
        self.btn_test.setProperty("class", "active" if index == 1 else "")
        # 强制刷新样式
        self.btn_train.style().polish(self.btn_train)
        self.btn_test.style().polish(self.btn_test)


class TrainingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.epoch_value = None
        self.script_path = None
        self.epoch = -1
        self.initUI()
        self.initPlots()
        self._is_running = True
        self.training_data = {'loss': [], 'f1': [], 'auc': []}
        self.setupConnections()

    def initUI(self):
        # 模型选择区域
        model_group = QGroupBox("模型配置")
        model_layout = QHBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(['efficientAD', 'fastflow', 'padim', 'patchcore', 'reversedistillation', 'GLASS'])
        self.script_btn = QPushButton("选择训练脚本文件")

        model_layout.addWidget(QLabel("选择训练模型或训练脚本"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.script_btn)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_group.setLayout(model_layout)

        # 参数输入区域
        param_group = QGroupBox("训练参数")
        param_layout = QGridLayout()
        param_layout.setHorizontalSpacing(10)  # 水平间距
        param_layout.setVerticalSpacing(5)

        # 第一行参数：训练轮次 和 批大小
        self.epochs = QLineEdit('100')
        self.batch_size = QLineEdit('32')
        param_layout.addWidget(QLabel("训练轮次:"), 0, 0, Qt.AlignRight)
        param_layout.addWidget(self.epochs, 0, 1)
        param_layout.addWidget(QLabel("批大小:"), 0, 3, Qt.AlignRight)
        param_layout.addWidget(self.batch_size, 0, 4)

        # 第二行参数：学习率 和 优化器
        self.lr = QLineEdit('0.001')
        self.optimizer = QComboBox()
        self.optimizer.addItems(['Adam', 'SGD', 'RMSprop'])
        param_layout.addWidget(QLabel("学习率:"), 1, 0, Qt.AlignRight)
        param_layout.addWidget(self.lr, 1, 1)
        param_layout.addWidget(QLabel("优化器:"), 1, 3, Qt.AlignRight)
        param_layout.addWidget(self.optimizer, 1, 4)

        # 第三行参数：权重衰减 和 验证集比例
        self.weight_decay = QLineEdit('0.0001')
        self.val_ratio = QLineEdit('0.2')
        param_layout.addWidget(QLabel("权重衰减:"), 2, 0, Qt.AlignRight)
        param_layout.addWidget(self.weight_decay, 2, 1)
        param_layout.addWidget(QLabel("验证集比例:"), 2, 3, Qt.AlignRight)
        param_layout.addWidget(self.val_ratio, 2, 4)

        # 数据集路径选择
        self.data_dir = QLineEdit()
        self.data_btn = QPushButton("浏览...")
        data_widget = QWidget()
        data_layout = QHBoxLayout()
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.addWidget(self.data_dir)
        data_layout.addWidget(self.data_btn)
        data_widget.setLayout(data_layout)

        # 输出路径选择
        self.output_path = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        output_widget = QWidget()
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(self.output_btn)
        output_widget.setLayout(output_layout)

        # 调整布局
        param_layout.addWidget(QLabel("数据集路径:"), 3, 0, Qt.AlignRight)
        param_layout.addWidget(data_widget, 3, 1, 1, 2)
        param_layout.addWidget(QLabel("输出路径:"), 3, 3, Qt.AlignRight)
        param_layout.addWidget(output_widget, 3, 4, 1, 2)

        param_group.setLayout(param_layout)

        # 日志区域
        log_group = QGroupBox("训练日志")
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)

        # 训练曲线显示区域
        Graphics_group = QGroupBox("训练曲线")
        self.plot_widget = pg.GraphicsLayoutWidget()
        Graphics_layout = QVBoxLayout()
        Graphics_layout.addWidget(self.plot_widget)
        Graphics_group.setLayout(Graphics_layout)

        # 控制按钮
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; height: 30px;")

        # 组装主界面
        layout = QVBoxLayout()

        layout.addWidget(QLabel("<h2>模型训练页面</h2>"))
        layout.addWidget(model_group)
        layout.addWidget(param_group)
        layout.addWidget(log_group)
        layout.addWidget(Graphics_group)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)

        # 美化样式
        self.setStyleSheet("""
                QGroupBox {
                    font: bold;
                    border: 1px solid silver;
                    margin-top: 10px;
                    padding: 10px;
                }
                QTextEdit {
                    background-color: #f0f0f0;
                }
                QPushButton {
                            background-color: #4CAF50;
                            border: 2px solid #3e643c;
                            border-radius: 15px;
                            color: white;
                            padding: 8px 16px;
                            font-size: 14px;
                            min-width: 80px;
                }
                QLineEdit, QComboBox {
                    max-width: 250px;
                    min-height: 25px;
                }
                QLabel {
                    min-width: 80px; /* 增加标签宽度以对齐更好 */
                }
            """)

    def initPlots(self):
        # 创建三个子图
        self.loss_plot = self.plot_widget.addPlot(title="Loss曲线")
        self.f1_plot = self.plot_widget.addPlot(title="F1 Score")
        self.auc_plot = self.plot_widget.addPlot(title="AUROC")
        self.plot_widget.nextRow()

        # 配置曲线样式
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen('r', width=2))
        self.f1_curve = self.f1_plot.plot(pen=pg.mkPen('g', width=2))
        self.auc_curve = self.auc_plot.plot(pen=pg.mkPen('b', width=2))

        # 设置图表样式
        for plot in [self.loss_plot, self.f1_plot, self.auc_plot]:
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', '数值')
            plot.setLabel('bottom', '训练轮次')
            plot.addLegend()

        self.loss_plot.setTitle("训练损失曲线", color='#FFF8DC', size='12pt')
        self.f1_plot.setTitle("F1 Score变化", color='#FFF8DC', size='12pt')
        self.auc_plot.setTitle("AUROC变化", color='#FFF8DC', size='12pt')
        # 设置纵轴范围为 0 到 1
        self.loss_plot.setYRange(0, 15)
        self.f1_plot.setYRange(0, 1)
        self.auc_plot.setYRange(0, 1)


    def updatePlots(self):
        # 更新曲线数据
        x_loss = np.arange(len(self.training_data['loss']))
        try:
            loss_data = [loss for loss in self.training_data['loss'] if loss is not None]
            self.loss_curve.setData(x_loss, np.array(loss_data, dtype=np.float64))
        except TypeError as e:
            if not isinstance(self.training_data['loss'], np.ndarray):
                data = np.array(self.training_data['loss'], dtype=np.float64)
                data = data[np.isfinite(data)]
                self.loss_curve.setData(x_loss, data)
        x_f1 = np.arange(len(self.training_data['f1']))
        f1_data = [f1 for f1 in self.training_data['f1'] if f1 is not None]
        self.f1_curve.setData(x_f1, np.array(f1_data, dtype=np.float64))
        x_auc = np.arange(len(self.training_data['auc']))
        auc_data = [auc for auc in self.training_data['auc'] if auc is not None]
        self.auc_curve.setData(x_auc, np.array(auc_data, dtype=np.float64))


    def setupConnections(self):
        self.data_btn.clicked.connect(self.selectDataDir)
        self.output_btn.clicked.connect(self.selectOutputDir)
        self.script_btn.clicked.connect(self.selectScriptFile)
        self.start_btn.clicked.connect(self.startTraining)

    def selectDataDir(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if path:
            self.data_dir.setText(path)

    def selectOutputDir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_path.setText(path)

    def selectScriptFile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型脚本", "", "Python Files (*.py)")
        if path:
            self.model_combo.addItem(os.path.basename(path))
            self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
            self.script_path = path

    def startTraining(self):
        if not self._is_running:
            self.log_output.append("任务正在进行中，请稍后。")
            return
        params = {
            'model': self.script_path,
            'epochs': self.epochs.text(),
            'batch_size': self.batch_size.text(),
            'learning_rate': self.lr.text(),
            'data_dir': self.data_dir.text(),
            'output_dir': self.output_path.text()
        }
        self.log_output.append("开始训练...")
        self.log_output.append(f"参数配置: {params}")
        # 获取 batch_size 的整数值
        try:
            self.epoch_value = int(params['epochs'])
        except ValueError:
            QMessageBox.warning(self, "错误", "批大小必须是整数！")

        # 参数校验
        if not all(params.values()):
            QMessageBox.warning(self, "错误", "请填写所有必填项！")
            return
        self._is_running = False
        self.updateButtonState(True)
        # 设置横轴范围为 0 到 batch_size
        for plot in [self.loss_plot, self.f1_plot, self.auc_plot]:
            plot.setXRange(0, self.epoch_value)
        command = [params['model'], '--dataset_root', params['data_dir'],'--max_epochs', params['epochs'],'--output_dir',params['output_dir']]
        # 启动训练进程
        self.process = QProcess()
        python_exe = sys.executable
        self.process.start(python_exe, command)
        # 连接输出和错误信号
        self.process.readyReadStandardOutput.connect(self.readOutput)
        self.process.readyReadStandardError.connect(self.readError)
        self.process.finished.connect(self.processFinished)

    def remove_ansi_escape_sequences(self, text):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)

    def clean_output(self, text):
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    def readOutput(self):
        raw_data = self.process.readAllStandardOutput().data()
        try:
            output = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            output = raw_data.decode('utf-8', errors='replace')
        output = self.remove_ansi_escape_sequences(output)
        output = self.clean_output(output)
        self.log_output.append(output)
        loss_pattern = re.compile(r"train_loss_epoch=(\d+\.\d+)")
        auc_pattern = re.compile(r"image_AUROC=(\d+\.\d+)")
        f1_pattern = re.compile(r"image_F1Score=(\d+\.\d+)")
        for line in output.splitlines():
            if "Epoch" in line:
                match = re.search(r'Epoch (\d+)', line)
                self.epoch = self.epoch + 1
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
                # 更新图表
                self.updatePlots()
                QApplication.processEvents()

    def updateButtonState(self, running):
        if running:
            self.start_btn.setText("模型正在训练,请稍后")
            self.start_btn.setStyleSheet("""
                background-color: #f44336;
                color: white;
                height: 35px;
                border-radius: 4px;
            """)
        else:
            self.start_btn.setText("开始训练")
            self.start_btn.setStyleSheet("""
                background-color: #4CAF50;
                color: white;
                height: 35px;
                border-radius: 4px;
            """)
    def readError(self):
        error = self.process.readAllStandardError().data()
        try:
            error = error.decode('utf-8', errors='replace').strip()
        except Exception as e:
            error = f"[Decoding Error] {str(e)}"
        error = self.remove_ansi_escape_sequences(error)
        error = self.clean_output(error)
        if error:
            self.log_output.append(f"{error}")

    def savePlot(plot, name):
        exporter = pg.exporters.ImageExporter(plot)
        exporter.export(f"{name}.png")


    def processFinished(self, exitCode, exitStatus):
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            self.log_output.append("训练完成。")
            self._is_running = True
            self.updateButtonState(False)
        else:
            self._is_running = True
            self.updateButtonState(False)
            self.log_output.append(f"训练失败，退出码: {exitCode}")




class TestingPage(QWidget):
    def __init__(self):
        super().__init__()
        self.process = None
        self.params = None
        self._is_running = True
        self.settings = QSettings("MyCompany", "MyAIApp")  # 初始化配置存储
        self.initUI()
        self.setupConnections()

    def initUI(self):
        layout = QVBoxLayout()

        # 配置区域
        config_group = QGroupBox("测试配置")
        config_layout = QFormLayout()

        # 模型路径选择
        self.model_path = QLineEdit()
        self.model_btn = QPushButton("浏览...")
        model_row = self.createFileRow(self.model_path, self.model_btn)

        # 配置文件选择
        self.config_path = QLineEdit()
        self.config_btn = QPushButton("浏览...")
        config_row = self.createFileRow(self.config_path, self.config_btn)

        # 测试数据选择
        self.test_data_path = QLineEdit()
        self.file_btn = QPushButton("选择文件")
        self.folder_btn = QPushButton("选择文件夹")
        data_row = self.createDataRow()

        # 测试脚本选择
        self.test_script_path = QLineEdit()
        self.script_btn = QPushButton("浏览...")
        script_row = self.createFileRow(self.test_script_path, self.script_btn)

        # 输出路径选择
        self.output_path = QLineEdit()
        self.output_btn = QPushButton("浏览...")
        output_row = self.createFileRow(self.output_path, self.output_btn)

        # 添加到布局
        config_layout.addRow(QLabel("模型路径:"), model_row)
        config_layout.addRow(QLabel("配置文件:"), config_row)
        config_layout.addRow(QLabel("测试数据:"), data_row)
        config_layout.addRow(QLabel("测试脚本:"), script_row)
        config_layout.addRow(QLabel("输出路径:"), output_row)
        config_group.setLayout(config_layout)

        # 结果展示区域
        result_group = QGroupBox("结果展示👈👉日志输出")
        result_layout = QHBoxLayout()

        # 图片显示区域
        self.image_label = QLabel()
        self.image_label.setStyleSheet("""
            border: 2px solid #333;
        """)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.log_output = QTextEdit()
        self.log_output.setStyleSheet("""
            border: 2px solid #333;
        """)
        self.log_output.setReadOnly(True)

        result_layout.addWidget(self.image_label, 3)
        result_layout.addWidget(self.log_output, 2)
        result_group.setLayout(result_layout)


        # 控制按钮
        self.test_btn = QPushButton("开始测试")
        self.test_btn.setStyleSheet("background-color: #4CAF50; color: white; height: 35px;")

        # 组装测试页面
        layout.addWidget(QLabel("<h2>模型测试页面</h2>"))
        layout.addWidget(config_group)
        layout.addWidget(result_group)
        layout.addWidget(self.test_btn)
        self.setLayout(layout)
        # 美化样式
        self.setStyleSheet("""
                        QGroupBox {
                            font: bold;
                            border: 1px solid silver;
                            margin-top: 10px;
                            padding: 10px;
                        }
                        QTextEdit {
                            background-color: #f0f0f0;
                        }
                        QPushButton {
                            background-color: #4CAF50;
                            border: 2px solid #3e643c;
                            border-radius: 15px;
                            color: white;
                            padding: 8px 16px;
                            font-size: 14px;
                            min-width: 80px;
                        }
                    """)

    def createFileRow(self, line_edit, button):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        widget.setLayout(layout)
        return widget

    def createDataRow(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.test_data_path)
        layout.addWidget(self.file_btn)
        layout.addWidget(self.folder_btn)
        widget.setLayout(layout)
        return widget

    def setupConnections(self):
        # 测试页面连接
        self.model_btn.clicked.connect(lambda: self.selectFile(
            self.model_path,
            "选择模型文件",
            "PyTorch Model (*.pth *.pt *.onnx)",
            "last_model_dir"
        ))
        self.config_btn.clicked.connect(lambda: self.selectFile(
            self.config_path,
            "选择配置文件",
            "JSON Files (*.json)",
            "last_config_dir"
        ))
        self.file_btn.clicked.connect(lambda: self.selectFile(
            self.test_data_path,
            "选择测试文件",
            "Images (*.png *.jpg *.jpeg)",
            "last_image_dir"
        ))
        self.folder_btn.clicked.connect(lambda: self.selectFolder(
            self.test_data_path,
            "last_dataset_dir"
        ))
        self.script_btn.clicked.connect(lambda: self.selectFile(
            self.test_script_path,
            "选择测试脚本",
            "Python Files (*.py)",
            "last_script_dir"
        ))
        self.output_btn.clicked.connect(lambda: self.selectFolder(self.output_path,"last_output_dir"))
        self.test_btn.clicked.connect(self.runTest)

    def selectFile(self, target, title, filter, setting_key):
        # 获取上次路径
        last_dir = self.settings.value(setting_key, QStandardPaths.writableLocation(QStandardPaths.HomeLocation))

        # 打开文件对话框
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            last_dir,  # 使用上次路径作为初始目录
            filter
        )

        if path:
            target.setText(path)
            # 保存本次路径的目录部分
            self.settings.setValue(setting_key, os.path.dirname(path))

    def selectFolder(self, target, setting_key="last_folder_dir"):
        last_dir = self.settings.value(setting_key, QStandardPaths.writableLocation(QStandardPaths.HomeLocation))

        path = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            last_dir
        )

        if path:
            target.setText(path)
            self.settings.setValue(setting_key, path)

    def runTest(self):
        # 获取参数
        if not self._is_running:
            self.log_output.append("任务正在进行中，请稍后。")
            return
        params = {
            "model_path": self.model_path.text(),
            "config_path": self.config_path.text(),
            "test_data": self.test_data_path.text(),
            "script_path": self.test_script_path.text(),
            "output_path": self.output_path.text()
        }

        # 参数校验
        if not all(params.values()):
            QMessageBox.warning(self, "错误", "请填写所有必填项！")
            return

        try:
            self._is_running = False
            self.updateButtonState(True)
            self.log_output.append("开始测试...")
            self.log_output.append(f"参数配置: {params}")
            command = [params['script_path'], '--path_model', params['model_path'], '--metadata', params['config_path'],'--path_dataset',params['test_data'],'--dir_result',params['output_path']]
            self.process = QProcess()
            python_exe = sys.executable
            self.process.start(python_exe, command)
            # 连接输出和错误信号
            self.process.readyReadStandardOutput.connect(self.readOutput)
            self.process.readyReadStandardError.connect(self.readError)
            self.process.finished.connect(self.processFinished)
            self.params = params

        except Exception as e:
            QMessageBox.critical(self, "错误", f"测试失败：{str(e)}")

    def remove_ansi_escape_sequences(self, text):
        ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)

    def clean_output(self, text):
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if line.strip()]
        return '\n'.join(cleaned_lines)

    def readOutput(self):
        output = self.process.readAllStandardOutput().data().decode()
        output = self.remove_ansi_escape_sequences(output)
        output = self.clean_output(output)
        self.log_output.append(output)

    def readError(self):
        error = self.process.readAllStandardError().data().decode('utf-8', errors='replace').strip()
        error = self.remove_ansi_escape_sequences(error)
        error = self.clean_output(error)
        if error:
            self.log_output.append(f"{error}")

    def processFinished(self, exitCode, exitStatus):
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            # 启动训练进程
            params = self.params
            name = self.folder_btn.text()
            if os.path.isdir(params['output_path']):
                image_path = os.path.join(params['output_path'], "confusion_matrix.png")
                if os.path.isfile(image_path):
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.image_label.setPixmap(pixmap.scaledToWidth(800))
                        self.log_output.append("图片加载完成。")
                    else:
                        self.log_output.append("图片加载失败，文件可能损坏或格式不支持。")
                else:
                    self.log_output.append("confusion_matrix.png 不存在于输出路径中。")
            else:
                self.log_output.append("输出路径不是一个有效的文件夹。")
            self.log_output.append("测试完成。")
            self._is_running = True
            self.updateButtonState(False)
        else:
            self._is_running = True
            self.updateButtonState(False)
            self.log_output.append(f"测试失败，退出码: {exitCode}")


    def updateButtonState(self, running):
        if running:
            self.test_btn.setText("正在测试,请稍后")
            self.test_btn.setStyleSheet("""
                background-color: #f44336;
                color: white;
                height: 35px;
                border-radius: 4px;
            """)
        else:
            self.test_btn.setText("开始测试")
            self.test_btn.setStyleSheet("""
                background-color: #4CAF50;
                color: white;
                height: 35px;
                border-radius: 4px;
            """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
