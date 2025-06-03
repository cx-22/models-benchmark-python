import cv2
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import re
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSpinBox,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSplitter,
    QSplitterHandle,
    QSlider,
    QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QStandardItem
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QSettings

import torch

release = False

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))

from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.models.experimental import attempt_load

logging.getLogger('ultralytics').setLevel(logging.WARNING)

def pad_to_multiple(image, multiple=32):
    h, w = image.shape[:2]
    
    new_h = (h // multiple + 1) * multiple
    new_w = (w // multiple + 1) * multiple
    
    pad_top = (new_h - h) // 2
    pad_bottom = new_h - h - pad_top
    pad_left = (new_w - w) // 2
    pad_right = new_w - w - pad_left
    
    return cv2.copyMakeBorder(
        image, pad_top, pad_bottom, pad_left, pad_right, 
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

class Worker(QThread):
    def __init__(self, win, frame1, frame2):
        super().__init__()
        self.window = win
        self.frame1 = frame1
        self.frame2 = frame2

    def run(self):
            if not self.frame2 is None:
                self.outframe = self.window.models[self.window.currentModel](self.frame1, self.frame2)
            else:
                self.outframe = self.window.models[self.window.currentModel](self.frame1, self.frame1)
            self.quit()

class DetachedWindow(QMainWindow):
    closed_signal = pyqtSignal()

    def __init__(self, lr):
        super().__init__()
        if lr == 0:
            self.setWindowTitle("Original")
        else:
            self.setWindowTitle("Processed")
        self.setGeometry(200, 200, 800, 600)

        stylesheet = """
                QPushButton {
                    padding: 3px;      /* Add padding around the text */
                    font-size: 25px;   /* Set font size */
                }
            """

        size_bar = QHBoxLayout()
        size_bar.setAlignment(Qt.AlignLeft)
        
        small_button = QPushButton("Smaller")
        small_button.clicked.connect(self.smaller)
        small_button.setStyleSheet(stylesheet)

        big_button = QPushButton("Bigger")
        big_button.clicked.connect(self.bigger)
        big_button.setStyleSheet(stylesheet)

        size_bar.addWidget(small_button, alignment=Qt.AlignLeft)
        size_bar.addWidget(big_button, alignment=Qt.AlignLeft)

        self.pixmap = None
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        main_layout = QVBoxLayout()
        main_layout.addLayout(size_bar)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def smaller(self):
        new_width = int(self.width() * 0.7)
        new_height = int(self.height() * 0.7)
        self.resize(new_width, new_height)
        if self.pixmap is not None:
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)

    def bigger(self):
        new_width = int(self.width() * 1.3)
        new_height = int(self.height() * 1.3)
        self.resize(new_width, new_height)
        if self.pixmap is not None:
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)


    def display(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.closed_signal.emit()
        event.accept()

class CheckableComboBox(QComboBox):
    def __init__(self, default):
        super().__init__()
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.closeOnLineEditClick = False
        self.lineEdit().installEventFilter(self)
        self.view().viewport().installEventFilter(self)
        self.model().dataChanged.connect(self.updateLineEditField)
        self.default = default

    
    def eventFilter(self, widget, event):
        if widget == self.lineEdit():
            if event.type() == QEvent.MouseButtonRelease:
                if self.closeOnLineEditClick:
                    self.hidePopup()
                else:
                    self.showPopup()
                return True
            return super().eventFilter(widget, event)
        
        if widget == self.view().viewport():
            if event.type() == QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                item = self.model().item(index.row())

                if item.checkState() == Qt.Checked:
                    item.setCheckState(Qt.Unchecked)
                else:
                    item.setCheckState(Qt.Checked)
                return True
            return super().eventFilter(widget, event)
        
    def hidePopup(self):
        super().hidePopup()
        self.startTimer(100)

    def addItems(self, items, itemList = None):
        for index, text in enumerate(items):
            try:
                data = itemList[index]
            except (TypeError, IndexError):
                data = None
            self.addItem(text, data)
        
    def addItem(self, text, userData = None):
        item = QStandardItem()
        item.setText(text)
        if not userData is None:
            item.setData(userData)
        
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        if text in self.default:
            item.setData(Qt.Checked, Qt.CheckStateRole)
        else:
            item.setData(Qt.Unchecked, Qt.CheckStateRole)
        self.model().appendRow(item)
    
    def updateLineEditField(self):
        text_container = []
        for i in range(self.model().rowCount()):
            if self.model().item(i).checkState() == Qt.Checked:
                text_container.append(self.model().item(i).text())
        text_string = ', '.join(text_container)
        self.lineEdit().setText(text_string)

class BetterSplitter(QSplitter):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

        self.setHandleWidth(20)

    def createHandle(self):
        return BetterSplitterHandle(self.orientation(), self)

class BetterSplitterHandle(QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)

        if orientation == Qt.Horizontal:
            self.symbol = QLabel("||", self)
        else:
            self.symbol = QLabel("==", self)


        layout = QVBoxLayout(self)
        layout.addWidget(self.symbol)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

class EGraphicsView(QGraphicsView):
    fileDroppedGraphics = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        self.fileDroppedGraphics.emit(filePath)

class Scene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        event.accept()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

class ImageViewerWidget(QWidget):
    fileDropped = pyqtSignal(str)

    def __init__(self, pixmap1, pixmap2, window, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.win = window

        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        self.splitter = BetterSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        self.view1 = EGraphicsView()
        self.scene1 = Scene()

        self.view1.fileDroppedGraphics.connect(self.dropGraphics)

        self.view1.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view1.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.view1.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view1.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.view1.setScene(self.scene1)
        self.view1.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.view1)

        self.view2 = EGraphicsView()
        self.scene2 = Scene()
        self.view2.setScene(self.scene2)
        self.view2.setAlignment(Qt.AlignCenter)
        self.splitter.addWidget(self.view2)

        self.pixmap1 = pixmap1
        self.pixmap2 = pixmap2

        self.pixmapItem = None
        
        self.add_image_to_scene(self.view1, self.scene1, pixmap1, False)
        self.add_image_to_scene(self.view2, self.scene2, pixmap2, False)

    def add_image_to_scene(self, view, scene, pixmap, isLeft):
        pixmap_item = QGraphicsPixmapItem(pixmap)
        if isLeft:
            self.pixmapItem = pixmap_item
        pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        scene.clear()
        scene.addItem(pixmap_item)

        def resize_event():
            view_width = view.viewport().width()
            view_height = view.viewport().height()

            scaled_pixmap = pixmap.scaled(view_width, view_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pixmap_item.setPixmap(scaled_pixmap)

            x_pos = (view_width - scaled_pixmap.width()) / 2
            y_pos = (view_height - scaled_pixmap.height()) / 2
            pixmap_item.setPos(x_pos, y_pos)

            scene.setSceneRect(0, 0, view_width, view_height)

        view.resizeEvent = lambda event: resize_event()
        resize_event()
    
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 0.8

        if event.angleDelta().y() > 0:
            self.view1.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.view1.scale(zoom_out_factor, zoom_out_factor)
        

    def load_left_image(self, pixmap):
        self.add_image_to_scene(self.view1, self.scene1, pixmap, True)

    def load_right_image(self, pixmap):
        self.add_image_to_scene(self.view2, self.scene2, pixmap, False)
    
    def reset_size(self):
        total_height = self.splitter.height()
        splitter_position = total_height // 2
        self.splitter.setSizes([splitter_position, splitter_position])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        self.fileDropped.emit(filePath)
    
    def dropGraphics(self, filePath):
        self.fileDropped.emit(filePath)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.yolov8s = self.yolov8m = self.yolov8l = self.yolov5m = self.yolov11m = None

        if release:
            self.yolov8s = YOLO('_internal/models/yolov8s.pt')
            self.yolov8m = YOLO('_internal/models/yolov8m.pt')
            self.yolov8l = YOLO('_internal/models/yolov8l.pt')
            self.yolov5m = YOLO('_internal/models/yolov5m.pt')
            self.yolov11m = YOLO('_internal/models/yolo11m.pt')
        else:
            self.yolov8s = YOLO('models/yolov8s.pt')
            self.yolov8m = YOLO('models/yolov8m.pt')
            self.yolov8l = YOLO('models/yolov8l.pt')
            self.yolov5m = YOLO('models/yolov5m.pt')
            self.yolov11m = YOLO('models/yolo11m.pt')

        self.device = None

        hasGPU = False

        if torch.cuda.is_available():
            print("GPU Name:", torch.cuda.get_device_name(0))
            self.yolov8s.to('cuda')
            self.yolov8m.to('cuda')
            self.yolov8l.to('cuda')
            self.yolov5m.to('cuda')
            self.yolov11m.to('cuda')
            hasGPU = True
        else:
            print("CUDA is not available. Using CPU.")

        self.yolov7 = None

        if hasGPU:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if release:
            self.yolov7 = attempt_load('_internal/models/yolov7.pt', map_location=self.device)
        else:
            self.yolov7 = attempt_load('models/yolov7.pt', map_location=self.device)


        imgsz = 640
        stride = int(self.yolov7.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        self.showFullScreen()
        self.setAcceptDrops(True)

        self.sources_folder = None

        if release:
            self.sources_folder = os.path.join(os.getcwd(), '_internal/sources')
        else:
            self.sources_folder = os.path.join(os.getcwd(), 'sources')

        screen = QApplication.primaryScreen()
        self.screen_width = screen.size().width()
        self.screen_height = screen.size().height()

        self.um = True
        self.firstWidth = 0
        self.firstHeight = 0

        self.stylesheet = """
            QLabel, QPushButton, QSpinBox, QComboBox, QLineEdit {
                text-align: left;  /* Left justify text */
                padding: 3px;      /* Add padding around the text */
                font-size: 20px;   /* Set font size */
                }
            """

        self.cmap_pixmap = None

        if release:
            self.cmap_pixmap = QPixmap('_internal/assets/legend.png')
        else:
            self.cmap_pixmap = QPixmap('assets/legend.png')

        self.webcams = []
        self.workers = []
        self.frameGrab = True

        self.is_detached_left = False
        self.is_detached_right = False
        self.detach_left = None
        self.detach_right = None
        
        self.current_media = None
        self.currentFrame = None
        self.currentPreproccessedFrame = None
        self.currentOutFrame = None

        self.fullscreen = True
        self.paused = True
        self.showMiddle = False
        self.ogFPS = 30
        self.newFPS = 0
        self.counter = 0
        self.cap = None
        self.options = None
        self.video_url = None
        self.liveVideo = False

        self.num_frames = 0
        self.duration = 0
        self.frame_position = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.video_manage)
        self.delay = int(1000 / self.ogFPS)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.find_fps)

        self.targetKeys = list(self.yolov8m.names.keys())
        self.targetNames = list(self.yolov8m.names.values())
        self.target = 0
        self.targetList = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']
        self.currentTargetsString = ['']
        self.currentTargetsIndex = [0]
        
        self.targetsCheckBox = CheckableComboBox(['person'])
        self.targetsCheckBox.addItems(self.targetList)
        self.targetsButton = QPushButton('Apply')
        self.targetsButton.clicked.connect(self.set_target)
        self.clearGraphButton = QPushButton('Apply and Clear Graph')
        self.clearGraphButton.clicked.connect(self.set_target_and_clear)

        self.targetsCheckBox.setStyleSheet(self.stylesheet)
        self.targetsButton.setStyleSheet(self.stylesheet)
        self.clearGraphButton.setStyleSheet(self.stylesheet)

        self.pop = 0
        self.max_pop = 20
        self.conf = 0.3

        self.currentPreprocess = "None"
        self.currentModel = "Yolov5m"
        self.pastModel = ""
        self.firstCycle = True
        
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(0, 0, 0))
        self.plot_graph.setTitle(f"{self.currentModel} Number of targets found", color="b", size="20pt")
        styles = {"color": "blue", "font-size": "14pt"}
        self.plot_graph.setLabel("bottom", "seconds", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(0, self.max_pop)
        self.timeList = list(range(10))
        self.popList = [0] * 10

        axis_font = pg.Qt.QtGui.QFont()
        axis_font.setPointSize(15)
        self.plot_graph.getAxis("left").setStyle(tickFont=axis_font)
        self.plot_graph.getAxis("bottom").setStyle(tickFont=axis_font)

        self.line = self.plot_graph.plot(
            self.timeList,                              
            self.popList,
            pen = pen,
            symbol="+",
            symbolSize = 15,
            symbolBrush="b"
            )

        self.timerD = QtCore.QTimer()
        self.timerD.start()
        self.timerD.setInterval(1000)
        self.timerD.timeout.connect(self.update_plot)

        self.windowTitle = "qtv3.1"
        self.setAcceptDrops(True)

        self.menu_bar = QHBoxLayout()
        self.options = QHBoxLayout()
        self.options.setAlignment(Qt.AlignCenter)
        self.detach_bar = QHBoxLayout()
        self.io_bar = QVBoxLayout()
        self.video_controls = QHBoxLayout()
        self.graph_space = QHBoxLayout()

        self.mid_widget = QWidget()
        self.graph_widget = QWidget()

        self.og_label = QLabel("Original")
        self.og_label.setStyleSheet(self.stylesheet)
        self.processed_label = QLabel("Processed")
        self.processed_label.setStyleSheet(self.stylesheet)

        self.file_dropdown = QComboBox()
        self.file_dropdown.setStyleSheet(self.stylesheet)

        self.file_dropdown.addItem("File")
        self.file_dropdown.addItem("Load Image or Video")
        self.file_dropdown.addItem("Save Output as Image")
        self.file_dropdown.currentIndexChanged.connect(self.set_file_menu)

        self.webcam_label = QLabel("Webcams")
        self.webcam_label.setStyleSheet(self.stylesheet)

        self.webcam_dropdown = QComboBox()
        self.webcam_dropdown.setStyleSheet(self.stylesheet)

        self.webcam_dropdown.addItem("")

        self.webcam_dropdown.currentIndexChanged.connect(self.set_webcam)

        self.device = 0
        if hasGPU:
            self.device_button = QPushButton("Swap to CPU")
            self.device_button.clicked.connect(self.swap_devices)
        else:
            self.device_button = QPushButton("Using CPU")
        
        self.device_button.setStyleSheet(self.stylesheet)

        self.preset_label = QLabel("Preset Cameras")
        self.preset_label.setStyleSheet(self.stylesheet)

        self.preset_dropdown = QComboBox()
        self.preset_dropdown.setStyleSheet(self.stylesheet)

        self.ip_camera_presets = [
            ["GMT-8",
            "http://204.106.237.68:88/mjpg/1/video.mjpg"],
            ["GMT-6",
            "http://104.207.27.126:8080/mjpg/video.mjpg",
            "http://63.142.183.154:6103/mjpg/video.mjpg",
            "http://lfytsquarecam.nctc.com/axis-cgi/mjpg/video.cgi",
            "http://cam3.aerostich.com:8888/axis-cgi/mjpg/video.cgi",
            "http://166.247.40.1:81/mjpg/video.mjpg",
            "http://64.77.205.140:80/mjpg/video.mjpg",
            "http://165.234.182.103:80/mjpg/video.mjpg",
            "http://207.118.17.26:80/mjpg/video.mjpg"],
            ["GMT-5",
            "http://buic010-mediacam-camera2.bu.edu/image",
            "http://98.102.110.114:82/mjpg/video.mjpg"],
            ["GMT+1",
            "http://82.134.72.194/mjpg/video.mjpg",
            "http://85.13.14.79/cgi-bin/faststream.jpg",
            "http://vroomshoopwebcam.mine.nu/axis-cgi/mjpg/video.cgi",
            "http://kamera.wseip.edu.pl/axis-cgi/mjpg/video.cgi",
            "http://webcam.vhs-ehingen.de/cgi-bin/faststream.jpg",
            "http://77.106.164.66/mjpg/video.mjpg",
            "http://90.146.10.190/mjpg/video.mjpg",
            "http://webcam.anklam.de/axis-cgi/mjpg/video.cgi",
            "http://webcam.minden-wlan.de:10000/axis-cgi/mjpg/video.cgi",
            "https://arc.manlleu.cat:448/axis-cgi/mjpg/video.cgi",
            "http://87.54.229.102/mjpg/video.mjpg",
            "http://159.130.70.206/mjpg/video.mjpg",
            "http://90.146.10.190/mjpg/video.mjpg",
            "http://webcam1.vilhelmina.se/axis-cgi/mjpg/video.cgi",
            "https://webcam.sparkassenplatz.info/cgi-bin/faststream.jpg",
            "http://195.228.161.126:8080/mjpg/video.mjpg",
            "http://91.187.63.35:88/mjpg/video.mjpg",
            "http://83.91.176.170:80/mjpg/video.mjpg"],
            ["GMT+3",
            "http://cam1.infolink.ru/mjpg/video.mjpg"],
            ["Unknown",
            "https://webcam.schwaebischhall.de/mjpg/video.mjpg"],
        ]

        self.preset_dropdown.addItem("Live Thailand GMT+7")
        self.preset_dropdown.addItem("Live Tokyo(intersect) GMT+9")
        self.preset_dropdown.addItem("Live Tokyo(people) GMT+9")
        self.preset_dropdown.addItem("Live St. Petersburg GMT+3")
        self.preset_dropdown.addItem("Live Finland GMT+2")
        self.preset_dropdown.addItem("Live Dublin GMT")
        self.preset_dropdown.addItem("Live Texas GMT-6")    
        self.preset_dropdown.addItem("Live Ontario GMT-5")
        self.preset_dropdown.addItem("Live California GMT-8")
        self.preset_dropdown.addItem("Prerecorded Nascar")
        
        self.presets = {
            "Live California GMT-8": "https://www.youtube.com/live/PtChZ0D7tkE?si=ylVs8s5BlIH2_aU6",
            "Live Texas GMT-6": "https://www.youtube.com/live/otX-buqqS6Q?si=ulxJadK_wuVYe1SB",
            "Live Ontario GMT-5": "https://www.youtube.com/watch?v=EPKWu223XEg",
            "Live Finland GMT+2": "https://www.youtube.com/live/Cp4RRAEgpeU?si=xXJ9tIJUD17qD9zt",
            "Live Dublin GMT": "https://www.youtube.com/live/u4UZ4UvZXrg?si=A5FSMhUJjX0gY7Yb",
            "Live St. Petersburg GMT+3": "https://www.youtube.com/live/h1wly909BYw?si=Boe9gLUcLcp6Za55",
            "Live Tokyo GMT+9": "https://www.youtube.com/live/DjdUEyjx8GM?si=-umo4EzSyDXDNkqd",
            "Live Tokyo(intersect) GMT+9": "https://www.youtube.com/live/6dp-bvQ7RWo?si=WS2viqCbLl-NxNzE",
            "Live Tokyo(people) GMT+9": "https://www.youtube.com/live/DjdUEyjx8GM?si=-umo4EzSyDXDNkqd",
            "Live Thailand GMT+7": "https://www.youtube.com/live/Q71sLS8h9a4?si=bGNflTmuPwexNC2k",
            "Prerecorded Nascar": "https://youtu.be/HU7wIi3VriY?si=IVr5EqfsOnMgm9Ya",
        }

        count = 0
        for zone in self.ip_camera_presets:
            zone_name = zone[0]
            for i in range(1, len(zone)):
                preset_name = "IP Preset " + str(count) + " " + zone_name
                self.preset_dropdown.addItem(preset_name)
                self.presets[preset_name] = zone[i]
                count += 1
            
        self.preset_dropdown.addItem("")

        self.url_bar = QLineEdit() 
        self.url_bar.returnPressed.connect(lambda: self.open_url(self.url_bar.text(), True))
        self.url_bar.setPlaceholderText("https://www.youtube.com/live/Q71sLS8h9a4?si=bGNflTmuPwexNC2k")
        self.url_bar.setStyleSheet(self.stylesheet)

        self.model_label = QLabel("Models")
        self.model_label.setStyleSheet(self.stylesheet)

        self.model_dropdown = QComboBox()
        self.model_dropdown.setStyleSheet(self.stylesheet)

        self.mini_button = QPushButton("Minimize")
        self.mini_button.clicked.connect(self.minimize)
        self.mini_button.setStyleSheet(self.stylesheet)

        self.fullscreen_button = QPushButton("Exit Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet(self.stylesheet)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_qt)
        self.exit_button.setStyleSheet(self.stylesheet)

        self.model_dropdown.addItem("Yolov5m")
        self.model_dropdown.addItem("Yolov7m")
        self.model_dropdown.addItem("Yolov8s")
        self.model_dropdown.addItem("Yolov8m")
        self.model_dropdown.addItem("Yolov8l")
        self.model_dropdown.addItem("Yolov11m")

        self.menu_bar.addWidget(self.file_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.webcam_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.webcam_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.preset_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.preset_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.url_bar, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.model_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.model_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.mini_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.device_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.fullscreen_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.exit_button, alignment=Qt.AlignTop)
        
        self.preprocess_label = QLabel("Preprocess: ")
        self.preprocess_label.setStyleSheet(self.stylesheet)

        self.preprocess_dropdown = QComboBox()
        self.preprocess_dropdown.currentIndexChanged.connect(self.set_preprocess)
        self.preprocess_dropdown.setStyleSheet(self.stylesheet)

        self.preprocess_dropdown.addItem("None")
        self.preprocess_dropdown.addItem("Degrade 1")
        self.preprocess_dropdown.addItem("Degrade 2")
        self.preprocess_dropdown.addItem("Degrade 3")

        self.preprocesses = {
            "Degrade 1":    self.degrade_1,
            "Degrade 2":    self.degrade_2,
            "Degrade 3":    self.degrade_3
        }

        self.show_middle_button = QPushButton("Show Intermediate")
        self.show_middle_button.clicked.connect(self.toggle_show_middle)
        self.show_middle_button.setStyleSheet(self.stylesheet)

        self.options_label = QLabel("Target: ")
        self.options_label.setStyleSheet(self.stylesheet)

        self.cmap_left = QLabel("-")
        self.cmap_left.setStyleSheet(self.stylesheet)
        self.cmap_right = QLabel("+")
        self.cmap_right.setStyleSheet(self.stylesheet)
        self.cmap_key = QLabel("")
        self.cmap_key.setPixmap(self.cmap_pixmap)

        self.conf_label = QLabel("Confidence")
        self.conf_label.setStyleSheet(self.stylesheet)

        self.conf_spin = QSpinBox(self)
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(30)
        self.conf_spin.setSingleStep(1)
        self.conf_spin.valueChanged.connect(self.set_conf)
        self.conf_spin.setStyleSheet(self.stylesheet)
        
        self.model_dropdown.currentIndexChanged.connect(self.set_model)

        self.detach_button = QPushButton("Detach Windows")
        self.detach_button.clicked.connect(self.toggle_detach)
        self.detach_button.setStyleSheet(self.stylesheet)    

        self.reset_splitter_button = QPushButton("Reset Splitter")
        self.reset_splitter_button.clicked.connect(self.reset_splitter)
        self.reset_splitter_button.setStyleSheet(self.stylesheet)    

        self.detach_bar.addWidget(self.reset_splitter_button, alignment=Qt.AlignCenter)

        self.og_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)

        basePixmap = QPixmap(480, 640)
        basePixmap.fill(Qt.black)

        self.io_viewer = ImageViewerWidget(basePixmap, basePixmap, self)
        self.io_viewer.fileDropped.connect(self.open_file)

        self.io_viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.clear_button = QPushButton("Clear Graph")
        self.clear_button.clicked.connect(lambda: (self.close_graph(), self.show_graph()))
        self.clear_button.setStyleSheet(self.stylesheet)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet(self.stylesheet)

        self.b_frame_button = QPushButton("Frame Back")
        self.b_frame_button.clicked.connect(self.b_frame)
        self.b_frame_button.setStyleSheet(self.stylesheet)

        self.f_frame_button = QPushButton("Frame Forward")
        self.f_frame_button.clicked.connect(self.f_frame)
        self.f_frame_button.setStyleSheet(self.stylesheet)

        self.live_button = QPushButton("Live Catch Up")
        self.live_button.clicked.connect(self.live_catchup)
        self.live_button.setStyleSheet(self.stylesheet)

        self.frame_reset_button = QPushButton("Frame Reset")
        self.frame_reset_button.clicked.connect(self.frame_reset)        
        self.frame_reset_button.setStyleSheet(self.stylesheet)

        self.fps_in = QSpinBox(self)
        self.fps_in.setRange(1, 120)
        self.fps_in.setValue(30)
        self.fps_in.setSingleStep(1)
        self.fps_in.valueChanged.connect(self.set_fps_in)
        self.fps_in.setStyleSheet(self.stylesheet)

        self.fps_in_label = QLabel("Input FPS: ")
        self.fps_in_label.setStyleSheet(self.stylesheet)

        self.fps_out_label = QLabel("Output FPS: " + str(self.newFPS))
        self.fps_out_label.setStyleSheet(self.stylesheet)

        self.fps_ratio_label = QLabel("FPS Out/In: ")
        self.fps_ratio_label.setStyleSheet(self.stylesheet)

        self.video_slider = QSlider(Qt.Orientation.Horizontal, self)

        self.video_slider.setStyleSheet("""
                                            QSlider {
                                                min-height: 40px;
                                                max-height: 40px;
                                                background:rgb(192, 192, 192);
                                            }

                                            QSlider::groove:horizontal {
                                                border: 1px solid #262626;
                                                height: 5px;
                                                background: #393939;
                                                margin: 0 12px;
                                            }

                                            QSlider::handle:horizontal {
                                                background:rgb(255, 0, 0);
                                                border: 5px solidrgb(0, 0, 0);
                                                width: 60px;
                                                height: 20px;
                                                margin: -24px -12px;
                                            }
                                        """)


        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(100)
        self.video_slider.setTickInterval(5)
        self.video_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.video_slider.sliderPressed.connect(self.pause_slider)
        self.video_slider.sliderReleased.connect(self.resume_slider)
        self.video_slider.sliderReleased.connect(self.slider_changed)

        self.video_controls.addWidget(self.clear_button)
        self.video_controls.addWidget(self.live_button)
        self.video_controls.addWidget(self.frame_reset_button)
        self.video_controls.addWidget(self.fps_in_label)
        self.video_controls.addWidget(self.fps_in)
        self.video_controls.addWidget(self.fps_out_label)
        self.video_controls.addWidget(self.fps_ratio_label)
        self.video_controls.addWidget(self.pause_button)
        self.video_controls.addWidget(self.b_frame_button)
        self.video_controls.addWidget(self.f_frame_button)

        self.video_controls.setAlignment(Qt.AlignCenter)

        self.graph_space.addWidget(self.plot_graph)#, alignment=Qt.AlignBottom)

        self.io_bar.addLayout(self.options)
        self.io_bar.addLayout(self.detach_bar)
        self.io_bar.addWidget(self.io_viewer)
        self.io_bar.addWidget(self.video_slider)
        self.io_bar.addLayout(self.video_controls)

        self.mid_widget.setLayout(self.io_bar)
        self.graph_widget.setLayout(self.graph_space)

        self.splitter = BetterSplitter(Qt.Vertical)  # Horizontal splitter

        self.splitter.addWidget(self.mid_widget)
        self.splitter.addWidget(self.plot_graph)

        self.splitter.setStyleSheet("QSplitter::handle { background-color: #CCCCCC; }")  # Set handle color to black

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_bar)
        self.main_layout.addWidget(self.splitter)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.models = {
            "Yolov5m":   self.apply_yolo,
            "Yolov7m":   self.apply_yolov7m,
            "Yolov8s":   self.apply_yolo,
            "Yolov8m":   self.apply_yolo,
            "Yolov8l":   self.apply_yolo,
            "Yolov11m":  self.apply_yolo
        }

        self.video_slider.hide()
        self.preset_dropdown.currentIndexChanged.connect(self.load_preset)

        self.currentModel = self.model_dropdown.currentText()
        self.currentFileName = ""
        self.currentFileExt = ""
        self.load_preset()
        self.set_model()
        self.build_webcams()
        self.preprocess_dropdown.setCurrentText("Degrade 1")
        self.model_dropdown.setCurrentIndex(1)

    def build_webcams(self):
        '''
        for i in range(20):
            testCap = cv2.VideoCapture(i)
            if testCap.isOpened():
                self.webcams.append(i)
        '''
        for i in self.webcams:
            self.webcam_dropdown.addItem(str(i))    

    def set_webcam(self):
        if (self.webcam_dropdown.currentText() != ""):
            self.currentFileName = "webcam"
            self.cap = cv2.VideoCapture(int(self.webcam_dropdown.currentText()))
            if self.cap.isOpened:
                self.kill_workers()
                self.liveVideo = True
                self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                self.fps_in.setValue(self.firstFPS)
                self.current_media = "vid"
                self.paused = False
                self.pause_button.setText("Pause")
                self.timer.start(self.delay)
                self.fps_timer.start(1000)

                if "Yolo" in self.currentModel:
                    self.close_graph()
                    self.show_graph()
                    self.new_display(self.currentFrame, "right")

    def update_plot(self):
        self.timeList = self.timeList[1:]
        self.timeList.append(self.timeList[-1] + 1)
        self.popList = self.popList[1:]
        self.popList.append(self.pop)
        self.line.setData(self.timeList, self.popList)

        if max(self.popList) >= (0.75 * self.max_pop):
            self.max_pop = max(self.popList) + 5
            self.plot_graph.setYRange(0, self.max_pop)

    def show_graph(self):
        self.plot_graph.show()
        if not self.paused:
            self.timerD.start()

    def close_graph(self):
        self.plot_graph.hide()
        self.timerD.stop()
        self.timeList = [0] * 10
        self.popList = [0] * 10
        self.max_pop = 20
        self.plot_graph.setYRange(0, self.max_pop)

    def video_manage(self):
        ret, frame = self.cap.read()
        if ret:
            self.currentFrame = frame.copy()

            if not self.liveVideo:
                self.frame_position += 1  # Move to the next frame
                time_pos = self.frame_position / self.firstFPS  # Convert to seconds
                slider_value = int(time_pos * 100)  # Convert to the scaled slider value
                self.video_slider.setValue(slider_value)

            if not self.currentPreprocess == "None":
                self.currentPreproccessedFrame = self.preprocesses[self.currentPreprocess](frame)
            if self.showMiddle and not self.currentPreprocess == "None":
                self.new_display(self.currentPreproccessedFrame, "left")
            else:
                self.new_display(frame, "left")
            if self.is_detached_left:
                self.detach_left.display(frame)
            if self.frameGrab:

                pixmap_rect = self.io_viewer.pixmapItem.boundingRect()
                if pixmap_rect.width() != 0 and pixmap_rect.width() != 0:

                    visible_scene_rect = self.io_viewer.view1.mapToScene(self.io_viewer.view1.viewport().rect()).boundingRect()

                    pixmap_scene_rect = self.io_viewer.pixmapItem.sceneBoundingRect()
                    visible_rect = visible_scene_rect.intersected(pixmap_scene_rect)

                    mapped_top_left = self.io_viewer.pixmapItem.mapFromScene(visible_rect.topLeft())
                    mapped_bottom_right = self.io_viewer.pixmapItem.mapFromScene(visible_rect.bottomRight())

                    x = int(mapped_top_left.x())
                    y = int(mapped_top_left.y())
                    width = int(mapped_bottom_right.x() - mapped_top_left.x())
                    height = int(mapped_bottom_right.y() - mapped_top_left.y())
                    
                    scaling_factor_x = self.currentFrame.shape[1] / pixmap_rect.width()
                    scaling_factor_y = self.currentFrame.shape[0] / pixmap_rect.height()

                    x = int(x * scaling_factor_x)
                    y = int(y * scaling_factor_y)
                    width = int(width * scaling_factor_x)
                    height = int(height * scaling_factor_y)

                    if self.currentPreprocess == "None":
                        fake_frame = self.currentFrame[y:y + height, x:x + width]
                    else:
                        fake_frame = self.currentPreproccessedFrame[y:y + height, x:x + width]
                    
                    new_worker = Worker(self, fake_frame, frame[y:y + height, x:x + width])
                    new_worker.finished.connect(self.on_worker_finished)
                    self.workers.append(new_worker)
                    new_worker.start()
                
                else:
                    if self.currentPreprocess == "None":
                        fake_frame = self.currentFrame
                    else:
                        fake_frame = self.currentPreproccessedFrame
                    new_worker = Worker(self, fake_frame, frame)
                    new_worker.finished.connect(self.on_worker_finished)
                    self.workers.append(new_worker)
                    new_worker.start()

                self.frameGrab = False

        else:
            self.frame_position = 0
            self.video_slider.setValue(0)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def on_worker_finished(self):
        dead_worker = self.sender()
        self.frameGrab = True
        if self.current_media == "vid":
            self.currentOutFrame = dead_worker.outframe
            self.new_display(self.currentOutFrame, "right")
            if self.is_detached_right:
                self.detach_right.display(self.currentOutFrame)
            self.newFPS += 1

        if dead_worker in self.workers:
            try:
                self.workers.remove(dead_worker)
            except ValueError:
                pass
        dead_worker.deleteLater() 

    def kill_workers(self):
        if len(self.workers) != 0:
            for worker in self.workers:
                worker.wait()
                worker.deleteLater()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        if filePath:
            self.open_file(filePath)

    def load_preset(self):
        if self.preset_dropdown.currentText() != "":
            self.preset_url = self.presets[self.preset_dropdown.currentText()]
            self.url_bar.setText(self.preset_url)
            self.open_url(self.preset_url, False)
    
    def open_url(self, link, close_preset):
        if link != "":
            temp_live = True
            if "youtube.com" in link or "youtu.be/" in link:
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]/best[ext=mp4]',
                    'quiet': True,
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(link, download=False)
                    if info.get('is_live', False):
                        temp_live = True
                        self.video_slider.hide()
                    else:
                        temp_live = False
                        self.video_slider.hide()
                    self.video_url = info['url']
                    self.cap = cv2.VideoCapture(self.video_url)
                    video_title = info.get('title', 'Unknown Title')
                    illegals = r'[<>:"/\\|?*\0]'
                    temp = re.sub(illegals, '-', video_title)
                    self.currentFileName = temp.strip().strip('.')
            else:
                self.currentFileName = "urlvideo"
                self.cap = cv2.VideoCapture(link)
            if self.cap.isOpened:
                self.kill_workers()
                self.liveVideo = temp_live
                self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                self.fps_in.setValue(self.firstFPS)
                self.current_media = "vid"
                self.paused = False
                self.pause_button.setText("Pause")
                self.webcam_dropdown.setCurrentIndex(0)
                '''
                if not temp_live:
                    self.video_slider.show()
                    self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.duration = self.num_frames / self.firstFPS
                    self.frame_position = 0
                    self.video_slider.setMaximum(int(self.duration * 100))
                '''
                if close_preset:
                    self.preset_dropdown.setCurrentText("")
                self.timer.start(self.delay)
                self.fps_timer.start(1000)

                if "Yolo" in self.currentModel:
                    self.close_graph()
                    self.show_graph()
                    self.new_display(self.currentFrame, "right")

    def set_file_menu(self):
        if self.file_dropdown.currentText() == "Load Image or Video":
            self.load_file()
            self.file_dropdown.setCurrentText("File")
        if self.file_dropdown.currentText() == "Save Output as Image":
            self.save_file()
            self.file_dropdown.setCurrentText("File")

    def load_file(self):
        filePath, _ = QFileDialog.getOpenFileName(None, "Open File", self.sources_folder,
                                                       "All Files (*);;Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mov *.mkv)")

        if filePath:
            self.open_file(filePath)

    def open_file(self, filePath):
        if filePath.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            self.currentFileExt = os.path.splitext(filePath)[1]
            self.currentFileName = os.path.basename(filePath).split('.')[0]
            self.current_media = "img"
            self.timer.stop()
            self.fps_timer.stop()
            self.close_graph()
            self.cap = None
            frame = cv2.imread(filePath)
            if frame is not None:
                self.video_slider.hide()
                self.url_bar.setText("")
                self.preset_dropdown.setCurrentText("")
                self.currentFrame = frame
                self.convert_image()
        if filePath.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
            self.cap = cv2.VideoCapture(filePath)
            self.currentFile = filePath
            if self.cap.isOpened:
                self.kill_workers()
                self.liveVideo = False
                self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                self.fps_in.setValue(self.firstFPS)
                self.video_slider.show()
                self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.num_frames / self.firstFPS
                self.frame_position = 0
                self.video_slider.setMaximum(int(self.duration * 100))
                self.currentFileExt = os.path.splitext(filePath)[1]
                self.currentFileName = os.path.basename(filePath).split('.')[0]
                self.current_media = "vid"
                self.paused = False
                self.pause_button.setText("Pause")
                self.url_bar.setText("")
                self.preset_dropdown.setCurrentText("")
                self.webcam_dropdown.setCurrentIndex(0)

                self.timer.start(self.delay)
                self.fps_timer.start(1000)

                if "Yolo" in self.currentModel:
                    self.close_graph()
                    self.show_graph()
                    #self.display(self.currentFrame, self.processed_label)
                    self.new_display(self.currentFrame, "right")


    def save_file(self):
        if self.currentOutFrame is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save File",
                self.currentFileName + ".png" if self.current_media == "vid" else self.currentFileName + self.currentFileExt,
                "PNG Files (*.png);;All Files (*)" if self.current_media == "vid" else f"All Files (*{self.currentFileExt})"
            )

            if save_path:
                cv2.imwrite(save_path, self.currentOutFrame)
        
    def swap_devices(self):
        self.kill_workers()
        if self.device == 0:
            self.yolov8s.to('cpu')
            self.yolov8m.to('cpu')
            self.yolov8l.to('cpu')
            self.yolov5m.to('cpu')
            self.yolov11m.to('cpu')
            device = torch.device('cpu')

            if release:
                self.yolov7 = attempt_load('_internal/models/yolov7.pt', map_location=device)
            else:
                self.yolov7 = attempt_load('models/yolov7.pt', map_location=device)

            self.device = 1
            self.device_button.setText("Swap to GPU")
        else:
            self.yolov8s.to('cuda')
            self.yolov8m.to('cuda')
            self.yolov8l.to('cuda')
            self.yolov5m.to('cuda')
            self.yolov11m.to('cuda')
            device = torch.device('cuda')

            if release:
                self.yolov7 = attempt_load('_internal/models/yolov7.pt', map_location=device)
            else:
                self.yolov7 = attempt_load('models/yolov7.pt', map_location=device)

            self.device = 0
            self.device_button.setText("Swap to CPU")
        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

    def minimize(self):
        self.setWindowState(Qt.WindowMinimized)
        self.fullscreen = False
        self.fullscreen_button.setText("Enter Fullscreen")

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
            self.fullscreen = False
            self.fullscreen_button.setText("Enter Fullscreen")
        else:
            self.showFullScreen()
            self.fullscreen = True
            self.fullscreen_button.setText("Exit Fullscreen")
    
    def exit_qt(self):
        self.kill_workers()
        self.close()
        QtCore.QCoreApplication.quit()
        sys.exit(0)

    def convert_image(self):
        if self.currentFrame is not None:
            if not self.currentPreprocess == "None":
                self.currentPreproccessedFrame = self.preprocesses[self.currentPreprocess](self.currentFrame)

            visible_scene_rect = self.io_viewer.view1.mapToScene(self.io_viewer.view1.viewport().rect()).boundingRect()

            pixmap_scene_rect = self.io_viewer.pixmapItem.sceneBoundingRect()
            visible_rect = visible_scene_rect.intersected(pixmap_scene_rect)

            mapped_top_left = self.io_viewer.pixmapItem.mapFromScene(visible_rect.topLeft())
            mapped_bottom_right = self.io_viewer.pixmapItem.mapFromScene(visible_rect.bottomRight())

            x = int(mapped_top_left.x())
            y = int(mapped_top_left.y())
            width = int(mapped_bottom_right.x() - mapped_top_left.x())
            height = int(mapped_bottom_right.y() - mapped_top_left.y())

            pixmap_rect = self.io_viewer.pixmapItem.boundingRect()

            cropped_frame = None
            dirty_crop = None

            if pixmap_rect.width() != 0 and pixmap_rect.width() != 0:
                scaling_factor_x = self.currentFrame.shape[1] / pixmap_rect.width()
                scaling_factor_y = self.currentFrame.shape[0] / pixmap_rect.height()

                x = int(x * scaling_factor_x)
                y = int(y * scaling_factor_y)
                width = int(width * scaling_factor_x)
                height = int(height * scaling_factor_y)

                cropped_frame = self.currentFrame[y:y + height, x:x + width]

                if not self.currentPreprocess == "None":
                    dirty_crop = self.currentPreproccessedFrame[y:y + height, x:x + width]
            
            else:
                cropped_frame = self.currentFrame
                if not self.currentPreprocess == "None":
                    dirty_crop = self.currentPreproccessedFrame

            frame = cropped_frame.copy()
            if not self.currentPreprocess == "None":
                self.currentOutFrame = self.models[self.currentModel](dirty_crop, frame)
            else:
                self.currentOutFrame = self.models[self.currentModel](cropped_frame, frame)

            if self.showMiddle and not self.currentPreprocess == "None":
                self.new_display(self.currentPreproccessedFrame, "left")
                if self.is_detached_left:
                    self.detach_left.display(self.currentPreproccessedFrame)
            else:
                self.new_display(self.currentFrame, "left")
                if self.is_detached_left:
                    self.detach_left.display(self.currentFrame)
        
            self.new_display(self.currentOutFrame, "right")
            if self.is_detached_right:
                self.detach_right.display(self.currentOutFrame)
    
    def reset_splitter(self):
        self.io_viewer.reset_size()

    def toggle_detach(self):
        self.detach_left = DetachedWindow(0)
        self.detach_left.closed_signal.connect(self.on_detach_end_left)
        self.detach_left.show()

        self.detach_right = DetachedWindow(1)
        self.detach_right.closed_signal.connect(self.on_detach_end_right)
        self.detach_right.show()

        self.is_detached_left = True
        self.is_detached_right = True

        if self.current_media == "img":
            self.convert_image()

    def on_detach_end_left(self):
        self.is_detached_left = False

    def on_detach_end_right(self):
        self.is_detached_right = False

    def display(self, frame, label):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            if self.plot_graph.isVisible():
                scaled_pixmap = pixmap.scaled(int(self.size().width() * 0.48), int(self.size().height() * 0.5), QtCore.Qt.KeepAspectRatio)
            else:
                scaled_pixmap = pixmap.scaled(int(self.size().width() * 0.48), int(self.size().height() * 0.7), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)
    
    def new_display(self, frame, side):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            if side == "left":
                self.io_viewer.load_left_image(pixmap)
            else:
                self.io_viewer.load_right_image(pixmap)

    def clear_option_layout(self):
        try:
            for i in range(self.options.count()):
                item = self.options.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        except AttributeError:
            pass
    
    def set_model(self):
        if not self.firstCycle:
            self.pastModel = self.currentModel
        self.currentModel = self.model_dropdown.currentText()

        if ("Yolo" in self.currentModel):
            if ("Yolo" in self.pastModel):
                self.plot_graph.setTitle(f"{self.currentModel} Number of targets found", color="b", size="20pt")
            else:
                self.clear_option_layout()
                
                self.options.addWidget(self.preprocess_label)
                self.options.addWidget(self.preprocess_dropdown)
                self.options.addWidget(self.show_middle_button)
                self.options.addWidget(self.conf_label)
                self.options.addWidget(self.conf_spin)
                self.options.addWidget(self.options_label)
                self.options.addWidget(self.targetsCheckBox)
                self.options.addWidget(self.targetsButton)
                self.options.addWidget(self.clearGraphButton)
                self.options.addWidget(self.cmap_left)
                self.options.addWidget(self.cmap_key)
                self.options.addWidget(self.cmap_right)
                if self.current_media == "vid":
                    self.close_graph()
                    self.show_graph()

        else:
            self.clear_option_layout()

        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

        self.firstCycle = False
    
    def set_preprocess(self):
        self.currentPreprocess = self.preprocess_dropdown.currentText()
        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

    def set_target(self):
        self.currentTargetsString = [word.strip() for word in self.targetsCheckBox.lineEdit().text().split(',')]
        self.currentTargetsIndex = []
        if len(self.currentTargetsString) != 0:
            for name in self.currentTargetsString:
                self.currentTargetsIndex.append(self.targetKeys[self.targetNames.index(name)])
        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()
        
    def set_target_and_clear(self):
        self.set_target()
        self.close_graph()
        self.show_graph()
        
    def toggle_pause(self):
        if self.current_media == "vid":
            if self.paused:
                self.timer.start()
                self.timerD.start()
                self.paused = False
                self.pause_button.setText("Pause")
            else:
                self.timer.stop()
                self.timerD.stop()
                self.paused = True
                self.pause_button.setText("Play")

    
    def pause_slider(self):
        self.timer.stop()
        self.timerD.stop()
        self.paused = True
        self.pause_button.setText("Play")
    
    def resume_slider(self):
        self.timer.start()
        self.timerD.start()
        self.paused = False
        self.pause_button.setText("Pause")
    
    def slider_changed(self):
        time_pos = self.video_slider.value() / 100
        self.frame_position = int(time_pos * self.firstFPS)
        self.kill_workers()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_position)
    
    def toggle_show_middle(self):
        if self.showMiddle:
            self.showMiddle = False
            self.show_middle_button.setText("Show Preprocess")
        else:
            self.showMiddle = True
            self.show_middle_button.setText("Show Original")

        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

            
    def b_frame(self):
        if self.current_media == "vid" and self.paused and not self.liveVideo:
            self.kill_workers()
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos - 2))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def f_frame(self):
        if self.current_media == "vid" and self.paused and not self.liveVideo:
            self.kill_workers()
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def set_fps_in(self):
        self.ogFPS = self.fps_in.value()
        self.delay = int(1000 / self.ogFPS)
        if not self.paused:
            self.timer.stop()
        if not self.paused:
            self.timer.start(self.delay)
    
    def find_fps(self):
        if not self.paused:
            self.fps_out_label.setText("Output FPS: " + str(self.newFPS))
            self.fps_ratio_label.setText(f"FPS Out/In: {self.newFPS / self.ogFPS:.2f}")
            self.newFPS = 0

    def frame_reset(self):
        if self.current_media == "vid":
            self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
            self.fps_in.setValue(self.firstFPS)
        
    def live_catchup(self):
        if self.liveVideo:
            self.kill_workers()
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_url)

    def set_conf(self):
        self.conf = (self.conf_spin.value() / 100)
        if (self.current_media == "vid" and self.paused) or self.current_media == "img":
            self.convert_image()
    
    def apply_yolo(self, frameI, frameOut):
        frameIn = frameI.copy()
        if frameIn is not None:
            if self.currentModel == "Yolov8m":
                output = self.yolov8m(frameIn, conf=self.conf)
            if self.currentModel == "Yolov8s":
                output = self.yolov8s(frameIn, conf=self.conf)
            if self.currentModel == "Yolov8l":
                output = self.yolov8l(frameIn, conf=self.conf)
            if self.currentModel == "Yolov5m":
                output = self.yolov5m(frameIn, conf=self.conf)
            if self.currentModel == "Yolov11m":
                output = self.yolov11m(frameIn, conf=self.conf)


            boxes = output[0].boxes.xyxy.cpu().numpy()
            class_ids = output[0].boxes.cls.cpu().numpy()
            confidences = output[0].boxes.conf.cpu().numpy()
            
            xpos = int(frameIn.shape[1] * 0.8)
            ypos = int(frameIn.shape[0] - 40)
            scale = frameIn.shape[1] * 0.00078125
            thickness =  1 if frameIn.shape[1] < 720 else 4
            labelThickness = 1 if frameIn.shape[1] < 720 else 2

            pop = 0

            outputSize = len(boxes)
            for i in range(outputSize):
                if class_ids[i] in self.currentTargetsIndex:
                    pop += 1
                    x1, y1, x2, y2 = map(int, boxes[i])
                    confidence = confidences[i]

                    new_val = int((confidence - 1) * -255)
                    gray = np.full((1, 1), new_val, dtype=np.uint8)
                    colored = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)
                    b, g, r = colored[0, 0]

                    color = (int(b), int(g), int(r))

                    cv2.rectangle(frameOut, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(frameOut, f"{self.targetNames[int(class_ids[i])]} {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        scale * 0.5, color, labelThickness) 

            self.pop = pop
            cv2.putText(frameOut, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                        scale, (0, 0, 0), thickness + 2)    
            cv2.putText(frameOut, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                                scale, (255, 255, 255), thickness)
            return frameOut
    
    def apply_yolov7m(self, frameI, frameO):
        frameIn = frameI.copy()
        frameOut = frameO.copy()
        frameIn = pad_to_multiple(frameIn, 32)
        frameIn = frameIn[:, :, ::-1].transpose(2, 0, 1)

        if (self.device == 0):
            dev = torch.device("cuda:0")
            frameIn = torch.from_numpy(frameIn.copy()).to(dev)
        else:
            dev = torch.device("cpu")     # For CPU mode
            frameIn = torch.from_numpy(frameIn.copy()).to(dev)
        
        frameIn = frameIn.float() / 255.0
        if frameIn.ndimension() == 3:
            frameIn = frameIn.unsqueeze(0)

        with torch.no_grad():
            pred = self.yolov7(frameIn, augment=False)[0]
        pred = non_max_suppression(pred, self.conf, 0.45, agnostic=True)

        pop = 0
        xpos = int(frameOut.shape[1] * 0.8)
        ypos = int(frameOut.shape[0] - 40)

        scale = frameOut.shape[1] * 0.00078125
        thickness =  1 if frameOut.shape[1] < 720 else 4
        labelThickness = 1 if frameOut.shape[1] < 720 else 2

        for det in pred:

            det[:, :4] = scale_coords(frameIn.shape[2:], det[:, :4], frameOut.shape).round()
            for *xyxy, conf, cls in det:
                if int(cls) in self.currentTargetsIndex:
                    pop += 1
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2, confidence = x1.item(), y1.item(), x2.item(), y2.item(), conf.item()
                    

                    new_val = int((confidence - 1) * -255)
                    gray = np.full((1, 1), new_val, dtype=np.uint8)
                    colored = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)
                    b, g, r = colored[0, 0]
                    color = (int(b), int(g), int(r))

                    cv2.rectangle(frameOut, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

                    cv2.putText(frameOut, f"{self.targetNames[int(cls)]} {confidence:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    scale * 0.5, color, labelThickness) 
                    
        self.pop = pop
        cv2.putText(frameOut, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                                scale, (0, 0, 0), thickness + 2)    
        cv2.putText(frameOut, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                                scale, (255, 255, 255), thickness)
        return frameOut

    def apply_grayscale(self):
        return cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)

    def degrade_1(self, inFrame):
        frame = inFrame.copy()
        frame = (frame +1)/8 - 1
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    def degrade_2(self, inFrame):
        frame = inFrame.copy()
        frame = (frame +1)/8 + 126
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    
    def degrade_3(self, inFrame):
        frame = inFrame.copy()
        frame = (frame +1)/8 + 223
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = Window()
    myWindow.show()
    sys.exit(app.exec_())