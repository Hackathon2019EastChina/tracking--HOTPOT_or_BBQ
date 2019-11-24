from gui import backend_gui_bridge as bgbridge
import gui.util.image_converter as iconverter
import cv2
import gui.util.cv_rectutil as cvrect
import gui.util.cv_drawline as cvline
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def __init__(self):
        self._backend = None
        self._current_frame = None
        self._filepath = None
        self._stop = False
        self._areas = []
        self._lines = []
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(460, 504)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.grpVideo = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.grpVideo.sizePolicy().hasHeightForWidth())
        self.grpVideo.setSizePolicy(sizePolicy)
        self.grpVideo.setObjectName("grpVideo")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.grpVideo)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lblVideoCapture = QtWidgets.QLabel(self.grpVideo)
        self.lblVideoCapture.setStyleSheet("border: 1px solid black")
        self.lblVideoCapture.setText("")
        self.lblVideoCapture.setObjectName("lblVideoCapture")
        self.lblVideoCapture.setScaledContents(True)
        self.gridLayout_2.addWidget(self.lblVideoCapture, 0, 0, 1, 1)
        self.sldVideoProcess = QtWidgets.QSlider(self.grpVideo)
        self.sldVideoProcess.setEnabled(False)
        self.sldVideoProcess.setOrientation(QtCore.Qt.Horizontal)
        self.sldVideoProcess.setObjectName("sldVideoProcess")
        self.gridLayout_2.addWidget(self.sldVideoProcess, 1, 0, 1, 1)
        self.verticalLayout.addWidget(self.grpVideo)
        self.grpFunciton = QtWidgets.QGroupBox(self.centralwidget)
        self.grpFunciton.setObjectName("grpFunciton")
        self.gridLayout = QtWidgets.QGridLayout(self.grpFunciton)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnSelectVideo = QtWidgets.QPushButton(self.grpFunciton)
        self.btnSelectVideo.setObjectName("btnSelectVideo")
        self.horizontalLayout.addWidget(self.btnSelectVideo)
        self.btnStartTracking = QtWidgets.QPushButton(self.grpFunciton)
        self.btnStartTracking.setObjectName("btnStartTracking")
        self.horizontalLayout.addWidget(self.btnStartTracking)
        self.btnStopTracking = QtWidgets.QPushButton(self.grpFunciton)
        self.btnStopTracking.setObjectName("btnStopTracking")
        self.horizontalLayout.addWidget(self.btnStopTracking)
        self.btnSelectArea = QtWidgets.QPushButton(self.grpFunciton)
        self.btnSelectArea.setObjectName("btnSelectArea")
        self.horizontalLayout.addWidget(self.btnSelectArea)
        self.btnSelectLine = QtWidgets.QPushButton(self.grpFunciton)
        self.btnSelectLine.setObjectName("btnSelectLine")
        self.horizontalLayout.addWidget(self.btnSelectLine)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.txtLog = QtWidgets.QPlainTextEdit(self.grpFunciton)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txtLog.sizePolicy().hasHeightForWidth())
        self.txtLog.setSizePolicy(sizePolicy)
        self.txtLog.setMaximumSize(QtCore.QSize(16777215, 140))
        self.txtLog.setReadOnly(True)
        self.txtLog.setObjectName("txtLog")
        self.verticalLayout_2.addWidget(self.txtLog)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.grpFunciton)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        '''
        ---> 开始连接
        '''
        self.btnSelectVideo.clicked.connect(self.OnBtnSelectVideoClicked)
        self.btnSelectArea.clicked.connect(self.OnBtnSelectAreaClicked)
        self.btnSelectLine.clicked.connect(self.OnBtnSelectLineClicked)
        self.btnStartTracking.clicked.connect(self.OnBtnStartTrackingClicked)
        self.btnStopTracking.clicked.connect(self.OnBtnStopTrackingClicked)
        '''
        ---> 结束连接
        '''
        self.btnSelectArea.setEnabled(False)
        self.btnStartTracking.setEnabled(False)
        self.btnSelectLine.setEnabled(False)
        self.btnStopTracking.setEnabled(False)
        '''
        ---> 初始禁用结束
        '''
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "视频跟踪前端程序"))
        self.grpVideo.setTitle(_translate("MainWindow", "视频追踪"))
        self.grpFunciton.setTitle(_translate("MainWindow", "追踪功能"))
        self.btnSelectVideo.setText(_translate("MainWindow", "选择视频"))
        self.btnStartTracking.setText(_translate("MainWindow", "开始追踪"))
        self.btnStopTracking.setText(_translate("MainWindow", "停止追踪"))
        self.btnSelectArea.setText(_translate("MainWindow", "选择禁区"))
        self.btnSelectLine.setText(_translate("MainWindow", "选择绊线"))

    def SetTrackingFrame(self, cvimage):
        qpixmap = iconverter.cvimage_to_qpixmap(cvimage, self.lblVideoCapture.width() - 2, self.lblVideoCapture.height() - 2)
        self.lblVideoCapture.setPixmap(qpixmap)

    def InitPreview(self, filename):
        cap = cv2.VideoCapture(filename)
        self.sldVideoProcess.setMaximum(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sldVideoProcess.setValue(0)
        ret, frame = cap.read()
        self.SetTrackingFrame(frame)
        self._current_frame = frame
        cap.release()

    def SetTrackingLog(self, log=[]):
        self.txtLog.clear()
        logstr = ''
        for item in log:
            logstr = logstr + item + '\r\n'
        self.txtLog.setPlainText(logstr)

    def OnBtnSelectVideoClicked(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(None, '打开视频文件', '.', '视频文件 (*.avi *.mp4)', '*')
        self.InitPreview(filename)
        self._areas = []
        self._lines = []
        self._filepath = filename
        self.btnSelectArea.setEnabled(True)
        self.btnStartTracking.setEnabled(True)
        self.btnStopTracking.setEnabled(False)
        self.btnSelectVideo.setEnabled(True)
        self.btnSelectLine.setEnabled(True)

    def OnBtnStartTrackingClicked(self):
        # 在播放完或停止前前禁用绊线、禁区、选视频功能
        self.btnSelectArea.setEnabled(False)
        self.btnStartTracking.setEnabled(False)
        self.btnStopTracking.setEnabled(True)
        self.btnSelectVideo.setEnabled(False)
        self.btnSelectLine.setEnabled(False)
        self._backend = bgbridge.BackendGuiBridge(self._filepath, self._lines, self._areas)
        self._stop = False
        success, frame = self._backend.get_frame()
        while success and not self._stop:
            self.sldVideoProcess.setValue(self.sldVideoProcess.value() + 1)
            self.SetTrackingFrame(frame)
            success, frame = self._backend.get_frame()
            self.SetTrackingLog(self._backend.get_logs())
            cv2.waitKey(100)
        self.OnBtnStopTrackingClicked()

    def OnBtnStopTrackingClicked(self):
        # 启用绊线、禁区、选视频功能
        self.btnSelectArea.setEnabled(True)
        self.btnStartTracking.setEnabled(True)
        self.btnStopTracking.setEnabled(False)
        self.btnSelectVideo.setEnabled(True)
        self.btnSelectLine.setEnabled(False)
        self._stop = True
        self.txtLog.clear()

    def OnBtnSelectAreaClicked(self):
        rgb_frame = self._current_frame
        cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR, rgb_frame)
        self._areas = cv2.selectROIs('Select Area', self._current_frame, False)
        cv2.destroyWindow('Select Area')
        frame = cvrect.draw_rects(self._current_frame, self._areas, (0, 0, 255))
        frame = cvrect.draw_lines(frame, self._lines, (0, 0, 255))
        self.SetTrackingFrame(frame)

    def OnBtnSelectLineClicked(self):
        rgb_frame = self._current_frame
        cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR, rgb_frame)
        self._lines = cvline.selectLines('Select Line', self._current_frame)
        frame = cvrect.draw_rects(self._current_frame, self._areas, (0, 0, 255))
        frame = cvrect.draw_lines(frame, self._lines, (0, 0, 255))
        self.SetTrackingFrame(frame)
