(ad_env) user@user-System-Product-Name:~/yaojie/Image_Anomaly_Detection-main$ export DISPLAY=:99
(ad_env) user@user-System-Product-Name:~/yaojie/Image_Anomaly_Detection-main$ Xvfb :99 -screen 0 1024x768x24 &
[1] 3684069
(ad_env) user@user-System-Product-Name:~/yaojie/Image_Anomaly_Detection-main$ python InterfaceTrain.py
/home/user/yaojie/Image_Anomaly_Detection-main/InterfaceTrain.py:10: DeprecationWarning: sipPyTypeDict() is deprecated, the extension module should use sipPyTypeDictRef() instead
  class MainWindow(QMainWindow):
/home/user/yaojie/Image_Anomaly_Detection-main/InterfaceTrain.py:92: DeprecationWarning: sipPyTypeDict() is deprecated, the extension module should use sipPyTypeDictRef() instead
  class TrainingPage(QWidget):
/home/user/yaojie/Image_Anomaly_Detection-main/InterfaceTrain.py:438: DeprecationWarning: sipPyTypeDict() is deprecated, the extension module should use sipPyTypeDictRef() instead
  class TestingPage(QWidget):
