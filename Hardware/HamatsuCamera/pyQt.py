import cv2
import sys
from PyQt6.QtWidgets import  QPushButton, QWidget, QLabel, QApplication
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import time

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Stream'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def initUI(self):

        self.startbutton = QPushButton("start",self)
        self.startbutton.clicked.connect(self.update)
        self.startbutton.resize(self.width+100,25)
        self.startbutton.setStyleSheet("background-color : green")
        #layout.addWidget(self.startbutton)

        self.stopbutton = QPushButton("stop",self)
        self.stopbutton.clicked.connect(self.close)
        self.stopbutton.resize(self.width+100,25)
        self.stopbutton.move(0,25)
        self.stopbutton.setStyleSheet("background-color : red")
        
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.width+100, self.height + 100)
        
        # create a label
        self.label = QLabel(self)
        self.label.move(50, 50)
        self.label.resize(640, 480)
        self.show()

    def close(self):
        self.startbutton.setStyleSheet("background-color : green")
        exit(0)
    
    def update(self):
        self.startbutton.setStyleSheet("background-color : white")
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                self.changePixmap.emit(p)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec())