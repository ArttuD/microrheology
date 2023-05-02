from dcam import *
from dcamapi4 import DCAMERR, DCAM_PIXELTYPE, DCAM_IDPROP, DCAM_PROP, DCAMPROP
import sys
import cv2 
import numpy as np
import time
import argparse
import os

import ffmpeg
#from PyQt6.QtCore import pyqtSignal, Qt
#from PyQt6.QtGui import QImage



class camera:
    #changePixmap = pyqtSignal(QImage)

    def __init__(self,args):
        if Dcamapi.init() is not False:
            self.dcam = Dcam(0,DCAM_PIXELTYPE.MONO8)
            
            self.dcam.dev_open()
            isExist = os.path.exists(args.path)
            
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(args.path)
            
            self.timeout = 1000
            self.data = None
            self.color = DCAM_PIXELTYPE.MONO8
            self.closeFlag = False
            self.out_process = None
            self.outName = os.path.join(args.path,"recording.avi")
            self.outNameScan = os.path.join(args.path,"recording_scan.avi")
            self.RoiWidth = 2064
            self.RoiHeight = 2064
            self.timesteps = np.stack((0,0), axis = -1)
            self.exposure = 25e-3
            self.fps = 40

            self.InitSettings()

            self.height, self.width = self.get_Dim()

            self.threshold = None
            self.mode = None
            #self.Qt = ex
        
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
            Dcamapi.uninit()
            exit(0)

    def startLive(self):
        self.mode = 0
        self.allocateBuffer(3)
        print("Please, tune exposure time from the microscope")
        self.liveImage()

    def startZmeasurement(self):
        self.mode = 1
        self.threshold = 1000
        self.allocateBuffer(self.threshold)
        self.initVideo()
        
        return self.getFrame()

    def startmeasurement(self):
        self.mode = 2
        self.threshold = int(85/self.exposure)
        self.allocateBuffer(self.threshold)
        self.initVideo()
        
        return self.getFrame()


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

    def initVideo(self):
        if self.mode == 1:
            self.out_process = cv2.VideoWriter(self.outNameScan, cv2.VideoWriter_fourcc('M','J','P','G'), 40.0, (self.width,self.height), False)

        else:
            self.out_process = cv2.VideoWriter(self.outName, cv2.VideoWriter_fourcc('M','J','P','G'), 40.0, (self.width,self.height), False)
    
    def saveVideo(self,x):
        self.out_process.write(x)


    def eventChecker(self):
        """Return 0 if no events. Otherwise return 1 if image taken. Otherwise return 2 if buffer full"""        
        ret = self.dcam.wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY + DCAMWAIT_CAPEVENT.STOPPED + DCAMWAIT_CAPEVENT.CYCLEEND,self.timeout)
        if not ret:
            return 0        
        else:
            return 1 if ret & DCAMWAIT_CAPEVENT.FRAMEREADY == DCAMWAIT_CAPEVENT.FRAMEREADY else 2    
        
    def process_image(self,x):       
        frame = x[0]
        self.data = x[1]
        self.timesteps = np.concatenate((self.timesteps,np.stack((frame.timestamp.sec,frame.timestamp.microsec),axis = -1)),axis = 0)
        
        #print("seconds", frame.timestamp.sec, "diff ", frame.timestamp.microsec-prev)            
        #prev = frame.timestamp.microsec            
        
        #imax = np.amax(self.data)
        #if imax > 0:
        #    imul = int(256 / imax)
        #    self.data = self.data * imul            
    
        if (self.i%10 == 0):
            iWindowStatus = self.displayframe()
     
        
    def saveBuffer(self):
        for i in range(self.i):
            if i%100 == 0:
                print("saving", i, "/", self.threshold)
            x = self.dcam.buf_getframe(i,self.color)
            if x == False:
                break
            else:
                self.saveVideo(x[1])
        self.out_process.release()
    
    def getFrame(self):
        self.i = 0      
        self.cameraStatus = True    
        self.dcam.cap_start(bSequence=False)        
           
        if self.mode == 2:
            print("Aqcuiring", self.threshold ,"frames")
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL) 
        else:
            print("Aqcuiring ", self.threshold, "frames")
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)   
            
        while self.cameraStatus:
            status = self.eventChecker()
            if status==1:
                self.process_image(self.dcam.buf_getframe(self.i,self.color))
                self.i += 1                
                # plot           
            elif status==2:
                # camera stopped                
                self.cameraStatus = False
                print("Camera stopped imaging, saving the recorded video stream")
            else:
                print("No events. Something fishy going on.") 
        
        cv2.destroyWindow("frame")
        # save buffer        
        self.saveBuffer()
        np.save(os.path.join(os.path.split(self.outName)[0],"frameInfo"),self.timesteps)
        self.stop()


    def liveImage(self):
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        self.cameraStatus = True
        self.dcam.cap_start()
        
        while self.cameraStatus:
            status = self.eventChecker()
            if (status==1) | (status == 2):
                self.data = self.dcam.buf_getlastframedata(self.color)
                iWindowStatus = self.displayframe()
            else:
                dcamerr = self.dcam.lasterr()
                if dcamerr.is_timeout():
                    print('===: timeout')
                else:
                    print('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))
                    
        
        cv2.destroyWindow("frame")
        self.stop()

    def displayframe(self):
        if  (self.data.dtype == np.uint8) | (self.data.dtype == np.uint16):         
            cv2.imshow("frame", self.data)
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):  
                # if 'q' was pressed with the live window, close it                
                self.cameraStatus = False                
                key = None            
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

    def close(self):
        print("closing")
        time.sleep(1)
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
    
    #self, hPos,vPost, hSize, wsize, status=False
    h,w = camClass.get_Dim()

    
    if not args.no_stack:

        print("Please, tune exposure time from the microscope and press Q to start the measurements")
        camClass.startLive()
        time.sleep(2)
    
        input("Press Enter to continue perform Z-scan...")
        camClass.startZmeasurement()
        time.sleep(1)

    input("Press Enter to continue to measurements... ")
    camClass.startmeasurement()
    
    time.sleep(2)

    #
    camClass.close()
    


"""
 print("Changing readout speed success? ", camClass.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST))
    print("Changing pixel type success? ", camClass.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, c_double(1)))
    print("Pixel type ", camClass.get_Value(DCAM_IDPROP.IMAGE_PIXELTYPE))
    
    #camClass.showProperties()
    print("exposure time set 25s, success: ", camClass.set_Value(DCAM_IDPROP.EXPOSURETIME, 24.99/1000), "\nexposure ", camClass.get_Value(DCAM_IDPROP.EXPOSURETIME) )
    print("Setting framerate to 40fps, success:", camClass.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, 40), camClass.get_Value(DCAM_IDPROP.INTERNALFRAMERATE))
    def displayframe(self):
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
"""  

"""
        #self.out_process.stdin.write(self.data.tobytes())
self.out_process = (
    ffmpeg
    .input('pipe:', format='rawvideo',pix_fmt='gray', s='{}x{}'.format(self.width, self.height))
    .filter('fps', fps=10, round='up')
    .output(self.outNameScan, pix_fmt='pal8', loglevel="quiet") #gray
    .overwrite_output()
    .run_async(pipe_stdin=True))
self.out_process = (
    ffmpeg
    .input('pipe:', format='rawvideo',pix_fmt='gray', s='{}x{}'.format(self.width, self.height))
    .filter('fps', fps=40, round='up')
    .output(self.outName, pix_fmt='pal8', loglevel="quiet") #gray
    .overwrite_output()
    .run_async(pipe_stdin=True))
"""