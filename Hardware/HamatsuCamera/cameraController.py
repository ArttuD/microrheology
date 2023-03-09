from dcam import *
from dcamapi4 import DCAMERR, DCAM_PIXELTYPE, DCAM_IDPROP, DCAM_PROP, DCAMPROP
import sys
import cv2 
import numpy as np
import time
import argparse
import os

import ffmpeg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QImage

class camera:
    changePixmap = pyqtSignal(QImage)

    def __init__(self,args):
        if Dcamapi.init() is not False:
            self.dcam = Dcam(0,DCAM_PIXELTYPE.MONO8)
            self.dcam.dev_open()
            isExist = os.path.exists(args.path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(args.path)
            self.timeout = 1000
            self.iWindowStatus = 0
            self.data = None
            self.color = DCAM_PIXELTYPE.MONO8
            self.closeFlag = False
            self.out_process = None
            self.outName = os.path.join(args.path,"recording.avi")
            self.outNameScan = os.path.join(args.path,"recording_scan.avi")
            self.RoiWidth = 1544
            self.RoiHeight = 2064
            self.timesteps = np.stack((0,0), axis = -1)
            self.exposure = 25e-3
            self.fps = 40
            self.InitSettings()
            self.height, self.width = self.get_Dim()
            self.stackFlag = True
            #self.Qt = ex
        
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
            Dcamapi.uninit()
            exit(0)

    def startLive(self):
        self.allocateBuffer(3)
        print("Please, tune exposure time from the microscope")
        self.liveImage()

    def startZmeasurement(self):
        self.allocateBuffer(1000)
        self.initVideo(flag = True)
        return self.getFrame(flag = True)

    def startmeasurement(self):
        self.allocateBuffer(1000)
        self.initVideo(flag = False)
        return self.getFrame(flag = False)


    def get_Value(self,prop):
        return self.dcam.prop_getvalue(prop)

    def set_Value(self,prop,value):
        return self.dcam.prop_setvalue(prop,value)

    def set_queryValue(self,prop,value):
        return self.dcam.prop_queryvalue(prop,value)

    def get_Dim(self):
        return int(self.get_Value(DCAM_IDPROP.IMAGE_HEIGHT)),int(self.get_Value(DCAM_IDPROP.IMAGE_WIDTH))

    def allocateBuffer(self,size):
        try:
            self.dcam.buf_alloc(size)
            return True
        except:
            return False

    def initVideo(self, flag):
        if flag:
            self.out_process = cv2.VideoWriter(self.outNameScan, cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (self.width,self.height), False)
            """
            self.out_process = (
                ffmpeg
                .input('pipe:', format='rawvideo',pix_fmt='gray', s='{}x{}'.format(self.width, self.height))
                .filter('fps', fps=10, round='up')
                .output(self.outNameScan, pix_fmt='pal8', loglevel="quiet") #gray
                .overwrite_output()
                .run_async(pipe_stdin=True))
            """
        else:
            self.out_process = cv2.VideoWriter(self.outName, cv2.VideoWriter_fourcc('M','J','P','G'), 40.0, (self.width,self.height), False)
            """
            self.out_process = (
                ffmpeg
                .input('pipe:', format='rawvideo',pix_fmt='gray', s='{}x{}'.format(self.width, self.height))
                .filter('fps', fps=40, round='up')
                .output(self.outName, pix_fmt='pal8', loglevel="quiet") #gray
                .overwrite_output()
                .run_async(pipe_stdin=True))
            """
    
    def saveVideo(self):
        self.out_process.write(self.data.astype("uint8"))
        #self.out_process.stdin.write(self.data.tobytes())

    def getFrame(self, flag):
        self.dcam.cap_start()
        self.j=0
        self.i = 0
        self.iWindowStatus = 1
        self.closeFlag = False
        if flag:
            print("Aqcuiring 1000 frames (25s)")
            self.Threshold = 1000
        else:
            print("Aqcuiring ", int(85/(self.exposure)), "frames (85s)")
            self.Threshold = int(85/(self.exposure))
        while self.iWindowStatus >= 0:
            if self.dcam.wait_capevent_frameready(self.timeout) is not False:
                x = self.dcam.buf_getframe(self.i,self.color)
                if (self.j > self.Threshold) | (x == False) | self.closeFlag == True:
                    if flag:
                        print("Z-stack ready")
                    else:
                        print("Measurement ready")
                    break
                else:
                    self.data = x[1]
                    if not flag:
                        frameStamp = x[0]
                        self.timesteps = np.concatenate((self.timesteps,np.stack((frameStamp.timestamp.sec,frameStamp.timestamp.microsec),axis = -1)),axis = 0)

                    imax = np.amax(self.data)
                    if imax > 0:
                        imul = int(256 / imax)
                        self.data = self.data * imul
                    
                    self.saveVideo()
                    if self.i%20 == 0:
                        self.iWindowStatus = self.displayframe()
                    
                    self.i += 1
                    self.j += 1
                    if self.i == 1000:
                        self.i = 0
            else: 
                print("No frames received")
                break

        self.out_process.release()
        self.stop()
        if not flag:
            np.save(os.path.join(os.path.split(self.outName)[0],"frameInfo"),self.timesteps)
            self.close()
            return -1
        return 1

    def liveImage(self):
        #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        self.dcam.cap_start()
        
        while self.iWindowStatus >= 0:
            if self.closeFlag == True:
                break
            if self.dcam.wait_capevent_frameready(self.timeout) is not False:
                self.data = self.dcam.buf_getlastframedata(self.color)
                imax = np.amax(self.data)
                if imax > 0:
                    imul = int(256 / imax)
                    self.data = self.data * imul
                self.iWindowStatus = self.displayframe()
            else:
                dcamerr = self.dcam.lasterr()
                if dcamerr.is_timeout():
                    print('===: timeout')
                else:
                    print('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))
                    break
        self.stop()

    def displayframe(self):
        if self.iWindowStatus < 0:
            return -1  # Window is already closed.
        if (self.data.dtype == np.uint8) | (self.data.dtype == np.uint16):
            h, w, ch = self.data.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(self.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
            self.changePixmap.emit(p)
            return 1
        else:
            print('-NG: dcamtest_show_image(data) only support Numpy.uint16 data')
            return -1
    
    def set_Roi(self, hPos,vPost, hSize, wsize, status=False):
        parameterValue_full = np.array([0, 2304, 0, 2304])
        parameterValue = np.array([hPos, hSize, vPost, wsize])
        #firtst set to full 
        for count,prop in enumerate([DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE, DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE]):
            reply = self.set_Value(prop, parameterValue_full[count])
            #print(reply)
            if reply == False:
                return False
        print("inacitvate subarray",self.set_Value(DCAM_IDPROP.SUBARRAYMODE, 1))
        
        if status == True:
            #self.height = hSize
            #self.width = wsize
            #Then actual frame size
            for count,prop in enumerate([DCAM_IDPROP.SUBARRAYHPOS, DCAM_IDPROP.SUBARRAYHSIZE, DCAM_IDPROP.SUBARRAYVPOS, DCAM_IDPROP.SUBARRAYVSIZE]):
                reply = self.set_Value(prop, parameterValue[count])
                #print(reply)
                if reply == False:
                    return False
            print("activate subarray",self.set_Value(DCAM_IDPROP.SUBARRAYMODE, 2))


        return True

    def get_WH(self):
        return self.get_Value(DCAM_IDPROP.IMAGE_WIDTH), self.get_Value(DCAM_IDPROP.IMAGE_HEIGHT)

    def EditCloseFlag(self):
        self.closeFlag = True
    
    def stop(self):
        self.dcam.cap_stop()
        self.dcam.buf_release()
        self.closeFlag = False

    def close(self):
        print("closing")
        time.sleep(1)
        #self.dcam.__close_hdcamwait()
        self.dcam.dev_close()
        Dcamapi.uninit()
        print("closed")

    def showProperties(self):

        idprop = self.dcam.prop_getnextid(0)
        while idprop is not False:
            output = '0x{:08X}: '.format(idprop)

            propname = self.dcam.prop_getname(idprop)
            if propname is not False:
                output = output + propname

                print(output)
                idprop = self.dcam.prop_getnextid(idprop)
            else:
                print('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))

    def InitSettings(self): 
        print("Setting ROI, success ", self.set_Roi(int((2304-self.RoiWidth)/2/4)*4,int((2304-self.RoiHeight)/2/4)*4, int(self.RoiWidth/4)*4, int(self.RoiHeight/4)*4, True))
        print("Changing readout speed, success: ", self.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST))
        print("Changing pixel type, success: ", self.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, c_double(1)))
        print("exposure time set 25s, success: ", self.set_Value(DCAM_IDPROP.EXPOSURETIME, self.exposure), "\nexposure ", self.get_Value(DCAM_IDPROP.EXPOSURETIME) )
        print("Setting framerate to 40fps, success:", self.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, self.fps), "\nfps ", self.get_Value(DCAM_IDPROP.INTERNALFRAMERATE))
        

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", help="path and name of output video")
    argParser.add_argument("-s", "--no_stack", help="If you dont want to record and check exposure", action = "store_true")

    args = argParser.parse_args()

    camClass = camera(args)
    print("Changing readout speed success? ", camClass.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST))
    print("Changing pixel type success? ", camClass.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, c_double(1)))
    print("Pixel type ", camClass.get_Value(DCAM_IDPROP.IMAGE_PIXELTYPE))
    
    #camClass.showProperties()
    print("exposure time set 25s, success: ", camClass.set_Value(DCAM_IDPROP.EXPOSURETIME, 24.99/1000), "\nexposure ", camClass.get_Value(DCAM_IDPROP.EXPOSURETIME) )
    print("Setting framerate to 40fps, success:", camClass.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, 40), camClass.get_Value(DCAM_IDPROP.INTERNALFRAMERATE))
    
    #self, hPos,vPost, hSize, wsize, status=False
    h,w = camClass.get_Dim()

    
    if not args.not_stack:
        print("Allocating buffer ", camClass.allocateBuffer(3))
        print("Please, tune exposure time from the microscope and press Q to start the measurements")
        camClass.liveImage()
        time.sleep(2)

        #print("exposure time set 25s, success: ", camClass.set_Value(DCAM_IDPROP.EXPOSURETIME, 24.99/1000), "\nexposure ", camClass.get_Value(DCAM_IDPROP.EXPOSURETIME) )
        #print("Setting framerate to 40fps, success:", camClass.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, 40), camClass.get_Value(DCAM_IDPROP.INTERNALFRAMERATE))
        camClass.allocateBuffer(1000)
        camClass.initVideo(flag = True)
        input("Press Enter to continue perform Z-scan...")
        camClass.getFrame(flag = True)
        time.sleep(1)

    camClass.allocateBuffer(1000)
    camClass.initVideo(flag = False)
    input("Press Enter to continue to measurements... ")
    time.sleep(1)
    camClass.getFrame(flag = False)
    time.sleep(2)

    #
    camClass.close()
    



