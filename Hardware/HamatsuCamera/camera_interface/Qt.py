from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal

from PyQt6.QtWidgets import *
from PyQt6.QtGui import QImage, QPixmap

import multiprocessing as mp

import sys
import numpy as np
import cv2
import time
import threading
import os
from queue import Queue

from Camera import camera

class App(QWidget):

    process_signal = pyqtSignal(int) 

    def __init__(self, args):
        super().__init__()
        self.ctrl = {}
        self.ctrl['break'] = False
        
        self.args = args

        self.event_cam = threading.Event()

        #UI geometry
        self.left = 0; self.top = 0
        self.width = 900; self.height = 900

        #Flags
        self.process_flag = False

        #Init driver and signal pipe
        self.cam = camera(self.event_cam, self.ctrl, self.args)
        self.cam.changePixmap.connect(self.setImage)
        self.cam.print_str.connect(self.receive_cam_str)
        self.process_signal.connect(self.cam.close_event)

        #cfg GUI
        self.initUI()

        #self.cam.showProperties()


    def initUI(self):
        
        self.win = QWidget()
        self.styles = {"color": "#f00", "font-size": "20px"}

        self.win.resize(self.width,self.height)
        
        #Main layout
        self.vlayout = QVBoxLayout()

        #1st row: Buttoms 
        self.hbutton = QHBoxLayout()

        #Text
        self.htext = QHBoxLayout()

        #4th row: sliders, fields and labels
        self.hlayout = QHBoxLayout()

        self.hlabels = QHBoxLayout()

        self.accessory = QVBoxLayout()
        
        self.cfg_buttons() #cfg buttons

        #2nd row: field path 
        self.textLabel()

        self.cfg_image() #set label

        #Add final layout and display
        self.win.setLayout(self.vlayout)
        self.win.show()  


    def cfg_image(self):

        """
        Create label, add accesories and label to layout
        """

        self.label = QLabel(self)
        #self.label.resize(480, 480)
        self.set_black_screen()

        self.hlabels.addWidget(self.label)#QtCore.Qt.AlignmentFlag.AlignCenter

        self.hlabels.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.hlabels.setSpacing(100)
        self.vlayout.addLayout(self.hlabels,Qt.AlignmentFlag.AlignCenter) 


    def set_black_screen(self):

        background = np.zeros((2048, 2048))
        h, w = background.shape
        bytesPerLine = 1 * w
        convertToQtFormat = QImage(background,w, h, bytesPerLine, QImage.Format.Format_Grayscale8)
        p = convertToQtFormat.scaled(720, 720) #Qt.AspectRatioMode.KeepAspectRatio
        self.label.setPixmap(QPixmap(QPixmap.fromImage(p)))  


    def textLabel(self):
        """
        File save path
        """
        self.textField = QTextEdit(self.args.path)
        self.textField.setFixedSize(int(self.width/2),50)

        self.printLabel = QLabel("Print Field")
   
        self.printLabel.setFixedSize(int(self.width/2),50)
        self.printLabel.setStyleSheet("background-color: white")

        self.htext.addWidget(self.textField)
        self.htext.addWidget(self.printLabel)
        self.vlayout.addLayout(self.htext)

    def createAndCheckFolder(self,path):
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)


    def cfg_buttons(self):
        """
        Create and connect buttons, sliders, and check box
        """
        #Start measurement
        self.btnStart = QPushButton("Start measurement")
        self.btnStart.pressed.connect(self.start)
        self.btnStart.setStyleSheet("background-color : green")

        self.hbutton.addWidget(self.btnStart)

        #Calibration button and autotune checkbox
        self.stackBtn = QPushButton("Z -stack")
        self.stackBtn.pressed.connect(self.z_stack)
        self.stackBtn.setStyleSheet("background-color : green")

        self.hbutton.addWidget(self.stackBtn)
        
        #Stream camera
        self.streamBtn = QPushButton("Live")
        self.streamBtn.pressed.connect(self.livestream)
        self.streamBtn.setStyleSheet("background-color : green")
        self.hbutton.addWidget(self.streamBtn)

        #Stop measurement or tuning
        self.btnStop = QPushButton("stop")
        self.btnStop.pressed.connect(self.stop)
        self.btnStop.setStyleSheet("background-color : red")
        self.hbutton.addWidget(self.btnStop)

        #Close Gui
        btnShutDown = QPushButton("shutdown")
        btnShutDown.pressed.connect(self.shutDown)
        btnShutDown.setStyleSheet("background-color : red")
        self.hbutton.addWidget(btnShutDown)

        #add buttons to main
        self.vlayout.addLayout(self.hbutton)


    def livestream(self):
        """
        Start camera stream
        *** No problems
        """
        self.process_flag = True
        self.streamBtn.setStyleSheet("background-color : white")
        self.process = QtCore.QThread()
        self.cam.moveToThread(self.process)

        self.process.started.connect(self.cam.startLive) 
        self.process.start()
        


    def z_stack(self):
        """
        -Autotune - calibration of magnetic sensor
        -Manual - Kalibrate current sensor to match the input
        """

        self.createAndCheckFolder(self.textField.toPlainText())
        self.cam.path = self.textField.toPlainText()
        self.stackBtn.setStyleSheet("background-color : white")

        self.printLabel.setText("Z -stack started")
        self.process_flag = True

        self.process = QtCore.QThread()
        self.cam.moveToThread(self.process)
        self.process.started.connect(self.cam.startZ) 
        self.process.start()

    def start(self):
        """
        Start measurement
            -Fetch path
            -start current driver and camera
        """

        self.createAndCheckFolder(self.textField.toPlainText())
        self.cam.path = self.textField.toPlainText()

        self.stackBtn.setStyleSheet("background-color : white")
        self.printLabel.setText("Measurement started")

        self.process_flag = True

        self.process = QtCore.QThread()
        self.cam.moveToThread(self.process)
        self.process.started.connect(self.cam.start) 
        self.process.start()
        

    def stop(self):
        """
        Stop tuning, measurement or camera stream and reset Gui
        """
        self.btnStop.setStyleSheet("background-color : white")
        print("closing")
        if self.process_flag:
            if self.process.isFinished() == False:
                print("emiting!")
                #self.process_signal.emit(1)
                self.ctrl['break'] = True
                while self.ctrl['break']:
                    print("waiting: ", self.ctrl['break'])
                    time.sleep(1)
                self.process.terminate()
                self.process.wait()
                #self.process.wait()
                time.sleep(2)
                print("Qthread closed, continue")
            else:
                print("process done")
        else:
            print("nothing running is running")


        self.streamBtn.setStyleSheet("background-color : green")
        self.stackBtn.setStyleSheet("background-color : green")  
        self.btnStart.setStyleSheet("background-color : green")
        self.btnStop.setStyleSheet("background-color : red")
        self.ctrl['break'] = False
        self.process_flag = False

        self.printLabel.setText("Ready for the new round!\nPlease remember change the path")
        self.set_black_screen()
    
    def shutDown(self):
        """
        Close all
        """
        self.printLabel.setText("Shutting down")
        self.cam.close()
        exit(0)

    @pyqtSlot(np.ndarray)
    def setImage(self, image):
        """
        Image signal pipe
        """
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Create a QImage from the normalized image
        q_image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 1, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        p = pixmap.scaled(720, 720) 
        self.label.setPixmap(p)

    @pyqtSlot(str)
    def receive_cam_str(self,data_str):
        """
        Driver Communication channel
        """
        self.printLabel.setText(data_str)


def pymain(args):
    app = QtWidgets.QApplication(sys.argv)
    w = App(args)
    sys.exit(app.exec())

if __name__ == "__main__":
    pymain()

