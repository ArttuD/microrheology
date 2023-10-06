

from dcam import *
from dcamapi4 import DCAM_PIXELTYPE, DCAM_IDPROP, DCAMPROP
import numpy as np
import time
import argparse
import os
import datetime

import ffmpeg

from PyQt6.QtCore import Qt, pyqtSignal,QThread
from PyQt6.QtGui import QImage

from queue import Queue

from threading import Event, Thread

#from PyQt6.QtCore import pyqtSignal, Qt
#from PyQt6.QtGui import QImage



class Hamamatsu_spark(QThread, Thread):
    changePixmap = pyqtSignal(QImage)
    print_str = pyqtSignal(str)

    def __init__(self, event, args):
        super().__init__()
        if Dcamapi.init() is not False:
            self.dcam = Dcam(0,DCAM_PIXELTYPE.MONO16)
            
            self.dcam.dev_open()
            isExist = os.path.exists(args.path)

            self.event = event
            
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(args.path)
            
            self.timeout = 1000
            self.exposure = 25e-3
            self.fps = 40
            self.RoiWidth = 2064
            self.RoiHeight = 2064
            self.color = DCAM_PIXELTYPE.MONO16

            self.outName = os.path.join(args.path,"recording.avi")
            self.outNameScan = os.path.join(args.path,"recording_scan.avi")

            self.saving_que = Queue(maxsize=300)

            self.InitSettings()

            self.height, self.width = self.get_Dim()

            self.threshold = None
            self.mode = None
        
        else:
            print('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
            Dcamapi.uninit()
            #exit(0)

    def livestream(self):
        self.mode = 0
        self.allocateBuffer(3)
        self.print_str("Please, tune exposure time from the microscope")
        self.liveImage()
        self.stop()

    def record_stack(self):
        self.mode = 1
        self.threshold = 1000
        self.allocateBuffer(self.threshold)
        save_event = Event()
        self.save_thread = Thread(target=self.camera_saving, args=(save_event,))
        self.save_thread.start()
        
        self.getFrame()

        save_event.set()
        self.save_thread.join()
        save_event.clear()

        self.stop()

        return 1

    def camera_saving(self,event_saver):
        out = self.initVideo()

        while True:
            if (self.saving_que.empty() == False):
                frame = self.saving_que.get()
                frame = np.stack((frame.astype("uint8"),frame.astype("uint8"),frame.astype("uint8")), axis = -1)
                #out.write()
                out.stdin.write(frame)
            else:
                if not event_saver.is_set():
                    time.sleep(0.01)
                else:
                    break
        #out.release()
        out.stdin.close()
        out.wait()
        return 1

    def record_measurement(self):
        self.mode = 2
        self.threshold = int(85/self.exposure)
        self.allocateBuffer(self.threshold)
        save_event = Event()
        self.save_thread = Thread(target=self.camera_saving, args=(save_event,))
        self.save_thread.start()
        
        self.getFrame()
    
        save_event.set()

        self.save_thread.join()
        save_event.clear()
        self.stop()

        return 1

    def allocateBuffer(self,size):
        try:
            self.dcam.buf_alloc(size)
            return True
        except:
            return False

    def initVideo(self):
        

        if self.mode == 1:
            #self.out_process = cv2.VideoWriter(self.outNameScan, cv2.VideoWriter_fourcc('M','J','P','G'), 40.0, (self.width,self.height), False)
            out_name = os.path.join(self.path,'measurement_scan_{}.mp4'.format(datetime.date.today()))
            out = ( 
                ffmpeg 
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
            .format(self.width, self.height)) 
            .output(out_name, pix_fmt='yuv420p') .overwrite_output() 
            .run_async(pipe_stdin=True) 
            )
        else:
            #self.out_process = cv2.VideoWriter(self.outName, cv2.VideoWriter_fourcc('M','J','P','G'), 40.0, (self.width,self.height), False)
            out_name = os.path.join(self.path,'measurement_{}.mp4'.format(datetime.date.today()))
            out = ( 
                ffmpeg 
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
            .format(self.width, self.height)) 
            .output(out_name, pix_fmt='yuv420p') .overwrite_output() 
            .run_async(pipe_stdin=True) 
            )

        return out

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

        self.emitFrame()
        self.saving_que.put(frame)

        self.timesteps = np.concatenate((self.timesteps,np.stack((frame.timestamp.sec,frame.timestamp.microsec),axis = -1)),axis = 0)
    
    def emitFrame(self):
        """
        Size the frame and send to Qt
        """
        h, w = self.data.shape
        ch = 1                
        bytesPerLine = ch * w

        convertToQtFormat = QImage(self.frame, w, h, bytesPerLine, QImage.Format.Format_Grayscale16) #
        p = convertToQtFormat.scaled(720, 720)#Qt.AspectRatioMode.KeepAspectRatio
        self.changePixmap.emit(p)

    def getFrame(self):
        self.i = 0      
        self.cameraStatus = True    
        self.dcam.cap_start(bSequence=False)        
           
        if self.mode == 2:
            print("Aqcuiring", self.threshold ,"frames")
        else:
            print("Aqcuiring ", self.threshold, "frames")
            
        while self.cameraStatus:
            status = self.eventChecker()
            if (status==1) & (self.event == False):
                self.process_image(self.dcam.buf_getframe(self.i,self.color))
                self.i += 1                
                # plot           
            elif status==2:
                # camera stopped                
                self.cameraStatus = False
                print("Camera stopped imaging, saving the recorded video stream")
            else:
                print("No events. Something fishy going on.") 
        
        np.save(os.path.join(os.path.split(self.outName)[0],"frameInfo"),self.timesteps)


    def liveImage(self):

        self.cameraStatus = True
        self.dcam.cap_start()
        
        while self.cameraStatus:
            status = self.eventChecker()

            if (status==1) | (status == 2) | (self.event == False):
                self.data = self.dcam.buf_getlastframedata(self.color)
                self.emitFrame()
            else:
                self.cameraStatus = False
                dcamerr = self.dcam.lasterr()
                if dcamerr.is_timeout():
                    print('===: timeout')
                else:
                    print('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))

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


    def stop(self):
        self.dcam.cap_stop()
        self.dcam.buf_release()

    def close(self):
        self.print_str.emit("closing")
        time.sleep(1)
        self.dcam.dev_close()
        Dcamapi.uninit()
        self.print_str.emit("closed")

    def get_Value(self,prop):
        return self.dcam.prop_getvalue(prop)

    def set_Value(self,prop,value):
        return self.dcam.prop_setvalue(prop,value)

    def set_queryValue(self,prop,value):
        return self.dcam.prop_queryvalue(prop,value)

    def get_Dim(self):
        return int(self.get_Value(DCAM_IDPROP.IMAGE_HEIGHT)),int(self.get_Value(DCAM_IDPROP.IMAGE_WIDTH))


    def showProperties(self):

        idprop = self.dcam.prop_getnextid(0)
        while idprop is not False:
            output = '0x{:08X}: '.format(idprop)

            propname = self.dcam.prop_getname(idprop)
            if propname is not False:
                output = output + propname

                self.print_str.emit(output)
                time.sleep(2)
                idprop = self.dcam.prop_getnextid(idprop)
            else:
                self.print_str.emit('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
        else:
            self.print_str.emit('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))

    def InitSettings(self): 
        #self.print_str.emit("Setting ROI, success ", self.set_Roi(int((2304-self.RoiWidth)/2/4)*4,int((2304-self.RoiHeight)/2/4)*4, int(self.RoiWidth/4)*4, int(self.RoiHeight/4)*4, True))
        ret = True
        ret *= self.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST)
        ret *= self.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, c_double(1))
        ret *= self.set_Value(DCAM_IDPROP.EXPOSURETIME, self.exposure)
        ret *= self.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, self.fps)

        self.print_str("Config success:", ret)
        

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
    