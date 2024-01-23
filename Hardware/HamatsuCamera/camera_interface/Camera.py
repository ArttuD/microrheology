import sys
import cv2 
import numpy as np
import datetime
import time
import argparse
import sys
import os
import multiprocessing as mp

from PyQt6.QtCore import Qt, pyqtSignal,QThread, pyqtSlot as Slot
from PyQt6.QtGui import QImage

import ffmpeg
from dcam import *
from dcamapi4 import DCAMERR, DCAM_PIXELTYPE, DCAM_IDPROP, DCAM_PROP, DCAMPROP

class camera(QThread):

    changePixmap = pyqtSignal(np.ndarray)
    print_str = pyqtSignal(str) #self.print_str.emit(pos)

    def __init__(self, ctr, args):
        super().__init__()
        self.ctrl = ctr

        if Dcamapi.init():
            
            self.dcam = Dcam(0,DCAM_PIXELTYPE.MONO16)
            self.dcam.dev_open()
            isExist = os.path.exists(args.path)
            
            if not isExist:
                # Create a new directory because it does not exist
                self.print_str.emit("Creating folder", args.path)
                os.makedirs(args.path)
            
            self.timeout = 1000
            self.data = None
            self.color = DCAM_PIXELTYPE.MONO16 #Only option for Spark

            self.closeFlag = False
            self.out_process = None

            self.outName = os.path.join(args.path,"recording.avi")
            self.outNameScan = os.path.join(args.path,"recording_scan.avi")

            self.timesteps = np.stack((0,0), axis = -1)
            self.exposure = 25e-3#25e-3
            self.fps = 40

            self.InitSettings()

            self.height, self.width = self.get_Dim()
            self.ch = 1  
            self.bytesPerLine = self.ch*self.width
            self.print_str.emit("ROI:\nHeight: {}\nWidth: {}".format(self.height, self.width))

            self.threshold = None
            self.mode = None
            self.path = args.path

        else:
            #self.print_str.emit('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
            self.print_str.emit("Check cable and power button!")
            self.print_str.emit("And restart")
            Dcamapi.uninit()
            exit(0)

    def run(self):
        value = self.ctrl["mode"]
        if value == 1:
            _ = self.startLive()
        elif value == 2:
            _ = self.startZ()
        elif value == 3:
            _ = self.start()

    def InitSettings(self):        
        self.print_str.emit("Changing readout speed, success: {}".format(self.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST)))
        self.print_str.emit("pixel type B/W 16 bit:".format(self.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, 2)))
        self.print_str.emit("Color type BW success:".format(self.set_Value(DCAM_IDPROP.COLORTYPE, DCAMPROP.COLORTYPE.BW)))
        self.print_str.emit("Bits per channel 16:".format(self.set_Value(DCAM_IDPROP.BITSPERCHANNEL, DCAMPROP.BITSPERCHANNEL._16)))
        self.print_str.emit("exposure time set 25s, success: {}\nexposure {}".format(self.set_Value(DCAM_IDPROP.EXPOSURETIME, self.exposure), self.get_Value(DCAM_IDPROP.EXPOSURETIME)))
        self.print_str.emit("Setting framerate to 40fps, success: {}\nfps {}".format(self.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, self.fps), self.get_Value(DCAM_IDPROP.INTERNALFRAMERATE)))
        

    def camera_saving(self, event_saver, q, path, width, height, ending):
        self.print_str.emit("creating saver")
        out_name = os.path.join(path,'measurement_{}_{}.mp4'.format(ending, datetime.date.today()))
        f = open(os.path.join(path,'measurement_{}_{}.csv'.format(ending, datetime.date.today())),'w')
        f.write('index, mSec, uSec\n')
        
        out_process = ( 
        ffmpeg 
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
        .format(width, height)) 
        .output(out_name, pix_fmt='yuv420p') .overwrite_output() 
        .run_async(pipe_stdin=True) 
        )
        idx = 0

        while True:
            if (q.empty() == False):
                packet = q.get()
                f.write(idx, ',',  packet[0].timestamp.sec, ',', packet[0].timestamp.microsec,'\n')
                frame = np.stack((packet[1].astype("uint8"),packet[1].astype("uint8"),packet[1].astype("uint8")), axis = -1)
                out_process.stdin.write( frame.tobytes() )
                idx += 1
            else:
                if not event_saver.is_set():
                    self.print_str.emit("waiting")
                    time.sleep(0.01)
                else:
                    self.print_str.emit("closing")
                    break
        
        f.close()
        out_process.stdin.close()
        out_process.wait()

        return 1
    

    def get_Value(self,prop):
        return self.dcam.prop_getvalue(prop)

    def set_Value(self,prop,value):
        return self.dcam.prop_setvalue(prop,value)

    def set_queryValue(self,prop,value):
        return self.dcam.prop_queryvalue(prop,value)

    def get_Dim(self):
        return int(self.get_Value(DCAM_IDPROP.IMAGE_HEIGHT)),int(self.get_Value(DCAM_IDPROP.IMAGE_WIDTH))

    def allocateBuffer(self,size):
        ret = self.dcam.buf_alloc(size)
        if ret:
            self.print_str.emit("buffer allocation {}".format(ret))
            return True
        else:
            self.print_str.emit("Failed to allocate space!")
            return False

    def get_WH(self):
        return self.get_Value(DCAM_IDPROP.IMAGE_WIDTH), self.get_Value(DCAM_IDPROP.IMAGE_HEIGHT)

    def showProperties(self):

        idprop = self.dcam.prop_getnextid(0)
        while idprop is not False:
            output = '0x{:08X}: '.format(idprop)

            propname = self.dcam.prop_getname(idprop)
            if propname is not False:
                output = output + propname

                self.print_str.emit(output)
                idprop = self.dcam.prop_getnextid(idprop)
            else:
                self.print_str.emit('-NG: Dcam.dev_open() fails with error {}'.format(self.dcam.lasterr()))
        else:
            self.print_str.emit('-NG: Dcamapi.init() fails with error {}'.format(Dcamapi.lasterr()))
    
    def startLive(self):
        self.mode = 0
        
        ret = self.allocateBuffer(3)
        #print("Buffer allocation: ", ret)

        if ret == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            self.close()
            sys.exit(1)
        #else:
        _ = self.liveImage()
        self.dcam.buf_release()
        print("Closed Everything, returning")
        self.ctrl['break'] = False
        return 1

    def startZ(self):

        self.mode = 1
        self.threshold = 1000
        worked = self.allocateBuffer(self.threshold)

        if worked == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            self.close()
            sys.exit()
            
        q = mp.Queue()
        save_event = mp.Event()
        save_thread = mp.Process(target= self.camera_saving, args=(save_event, q, self.path, self.width, self.height, "scan"))
        save_thread.start()

        self.getFrame(q)
        
        save_event.set()
        save_thread.join()

        save_event.clear()

        self.dcam.buf_release()

    def start(self):
        self.mode = 2

        self.threshold = 3225
        worked = self.allocateBuffer(self.threshold)
        if worked == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            self.close()
            sys.exit()

        q = mp.Queue()

        save_event = mp.Event()
        save_thread = mp.Process(target= self.camera_saving, args=(save_event, q, self.path, self.width, self.height, "main"))
        save_thread.start()

        self.getFrame(q)
        
        save_event.set()
        save_thread.join()
        save_event.clear()

        self.dcam.buf_release()
    
    def displayframe(self, frame):
        
        self.changePixmap.emit(frame)
        return 1

    def eventChecker(self):
        """Return 0 if no events. Otherwise return 1 if image taken. Otherwise return 2 if buffer full"""        
        ret = self.dcam.wait_event(DCAMWAIT_CAPEVENT.FRAMEREADY + DCAMWAIT_CAPEVENT.STOPPED + DCAMWAIT_CAPEVENT.CYCLEEND,self.timeout)
        if not ret:
            return 0        
        else:
            return 1 if ret & DCAMWAIT_CAPEVENT.FRAMEREADY == DCAMWAIT_CAPEVENT.FRAMEREADY else 2    
        
    
    def liveImage(self):

        self.cameraStatus = True
        self.dcam.cap_start()
        while self.cameraStatus:
            status = self.eventChecker()
            if (status==1) & (self.ctrl['break'] == False):
                packet = self.dcam.buf_getlastframedata(self.color)
                iWindowStatus = self.displayframe(packet)
                
            else:
                if self.ctrl['break']:
                    print("received close command")
                    break
                elif status == 2:
                    continue
                else:
                    dcamerr = self.dcam.lasterr()
                    if dcamerr.is_timeout():
                        self.print_str.emit('===: timeout')
                    else:
                        self.print_str.emit('-NG: Dcam.wait_event() fails with error {}'.format(dcamerr))
                        
        self.dcam.cap_stop()
        return 0

    def getFrame(self, q):
        self.i = 0 #num_recovered images
        self.cameraStatus = True    
        self.dcam.cap_start(bSequence=False)        
           
        if self.mode == 2:
            self.print_str.emit("Aqcuiring", self.threshold ,"frames")
        else:
            self.print_str.emit("Aqcuiring ", self.threshold, "frames")
            
        while self.cameraStatus:
            status = self.eventChecker()

            if (self.ctrl['break'] == False) | (self.i == self.threshold):
                self.dcam.cap_stop()
            elif (status==1) &  (self.i < self.threshold): 
                packet = self.dcam.buf_getframe(self.i,self.color)
                if type(packet) == bool:
                    self.cameraStatus = False
                    self.dcam.cap_stop()
                    self.print_str.emit("Failed to recover image, closing after", self.i, "images!")
                q.put(packet)
                if self.i%5 == 0:
                    iWindowStatus = self.displayframe(packet[1])
                self.i += 1                
                # plot           
            elif status==2:
                # camera stopped
                if (self.ctrl['break'] == False) | (self.i == self.threshold):
                    self.cameraStatus = False
                    self.dcam.cap_stop()
                    self.print_str.emit("Camera done, recovered", self.i, "images!")
            else:
                self.print_str.emit("No events.", status, "Something fishy going on.", self.dcam.lasterr()) 
                self.dcam.cap_stop()
                self.cameraStatus = False
        
    def close(self):
        self.print_str.emit("closing Camera")
        self.dcam.cap_stop()
        self.dcam.buf_release()
        
        self.dcam.dev_close()
        Dcamapi.uninit()


class saverTest:
    def __init__(self):
        pass
        self.path = "C:/Users/lehtona6/Downloads/Image_895.tif"
        self.image = cv2.imread( self.path, cv2.IMREAD_UNCHANGED)
        self.q = mp.Queue()
        self.save_event = mp.Event()

    def camera_saving(self, event_saver, q, path,):
        out_name = os.path.join(path,'measurement_{}.mp4'.format(datetime.date.today()))

        out_process = ( 
        ffmpeg 
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'
        .format(2068, 1544)) 
        .output(out_name, pix_fmt='yuv420p') .overwrite_output() 
        .run_async(pipe_stdin=True) 
        )
        idx = 0

        while True:
            if (q.empty() == False):
                packet = q.get()
                frame = np.stack((packet.astype("uint8"),packet.astype("uint8"),packet.astype("uint8")), axis = -1)
                out_process.stdin.write( frame.tobytes() )
            else:
                if not event_saver.is_set():
                    time.sleep(0.01)
                else:
                    break
        
        out_process.stdin.close()
        out_process.wait()

        return 1
    
    def start_thread(self):

        self.save_thread = mp.Process(target= self.camera_saving, args=(self.save_event, self.q, os.path.split(self.path)[0]))
        self.save_thread.start()
    
    def main_pipe(self):
        try:
            self.start_thread()
        except:
            exit(0)
        for i in range(100):
            self.q.put(self.image)
            time.sleep(0.1)


        self.save_event.set()
        self.save_thread.join()
        self.save_event.clear()

if __name__ == '__main__':
    tester = saverTest()
    tester.main_pipe()
    
    """
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", help="path and name of output video")
    argParser.add_argument("-s", "--no_stack", help="If you dont want to record and check exposure", action = "store_true")
    args = argParser.parse_args()

    camClass = camera(args)

    if not args.no_stack:

        camClass.startLive()
        time.sleep(3)
    
        input("Press Enter to continue perform Z-scan...")
        camClass.startZmeasurement()
        time.sleep(3)

    input("Press Enter to continue to measurements... ")
    camClass.startmeasurement()
    
    time.sleep(3)

    #
    camClass.close()
    """

