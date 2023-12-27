from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtSlot
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

    def __init__(self, args):
        super().__init__()

        self.args = args

        self.event_cam = threading.Event()

        #UI geometry
        self.left = 0; self.top = 0
        self.width = 900; self.height = 900

        #Flags
        self.process_flag = False

        #Init driver and signal pipe
        self.cam = camera(self.event_cam, self.args)
        self.cam.changePixmap.connect(self.setImage)
        self.cam.print_str.connect(self.receive_cam_str)

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

        #Process

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
        print("Please, brightness and find sample")
        self.process_flag = True
        self.streamBtn.setStyleSheet("background-color : white")
        self.process_event = mp.Event()

        self.process = mp.Process(target= self.cam.startLive, args=(self.process_event))
        self.process.start()
        


    def z_stack(self):
        """
        -Autotune - calibration of magnetic sensor
        -Manual - Kalibrate current sensor to match the input
        ***Problems
        * Autotune, not tested!
        """
        path = self.textField.toPlainText()
        self.stackBtn.setStyleSheet("background-color : white")

        self.printLabel.setText("Z -stack started")
        self.process_flag = True
        self.createAndCheckFolder(path)
        self.process_event = mp.Event()
        self.process = mp.Process(target= self.cam.startZ, args=(self.process_event, path))
        self.process.start()

    def start(self):
        """
        Start measurement
            -Fetch path
            -start current driver and camera
        """
        path = self.textField.toPlainText()
        self.stackBtn.setStyleSheet("background-color : white")

        self.printLabel.setText("Measurement started")
        self.process_flag = True
        self.createAndCheckFolder(path)
        self.process_event = mp.Event()
        self.process = mp.Process(target= self.cam.start, args=(self.process_event, path))
        self.process.start()
        

    def stop(self):
        """
        Stop tuning, measurement or camera stream and reset Gui
        """
        self.btnStop.setStyleSheet("background-color : white")
        if self.process_flag:
            if self.process.is_alive():
                self.process_event.set()
            
            self.process.join()
            self.process_event.clear()
            
            self.set_black_screen()
        else:
            print("nothing running is running")

        self.streamBtn.setStyleSheet("background-color : green")
        self.stackBtn.setStyleSheet("background-color : green")  
        self.btnStart.setStyleSheet("background-color : green")
        self.btnStop.setStyleSheet("background-color : red")

        self.printLabel.setText("Ready for the new round!\nPlease remember change the path")
    
    def shutDown(self):
        """
        Close all
        """
        self.printLabel.setText("Shutting down")
        self.cam.close()
        exit(0)

    @pyqtSlot(QImage)
    def setImage(self, image):
        """
        Image signal pipe
        """
        self.receivedFrame = image

        if (self.liveFlag == False) & (self.trackFlag == True) & (self.snapFlag == False):

            self.drawRectangle(QPixmap(QPixmap.fromImage(self.receivedFrame)))
        else:
            self.label.setPixmap(QPixmap(QPixmap.fromImage(self.receivedFrame)))

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

