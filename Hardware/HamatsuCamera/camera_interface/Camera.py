import sys
import numpy as np
import datetime
import time
import sys
import os
import multiprocessing as mp

from PyQt6.QtCore import Qt, pyqtSignal,QThread, pyqtSlot as Slot


from dcam import *
from dcamapi4 import DCAMERR, DCAM_PIXELTYPE, DCAM_IDPROP, DCAM_PROP, DCAMPROP


class camera(QThread):

    changePixmap = pyqtSignal(np.ndarray) #Image
    print_str = pyqtSignal(str) #Image
    

    def __init__(self, ctr, args):
        super().__init__()

        self.ctrl = ctr
        self.path = args.path

        if Dcamapi.init():
            
            self.dcam = Dcam(0,DCAM_PIXELTYPE.MONO16)
            self.dcam.dev_open()
            
            self.timeout = 1000
            self.color = DCAM_PIXELTYPE.MONO16 #Only option for Spark

            self.outName = os.path.join(args.path,"recording.avi")
            self.outNameScan = os.path.join(args.path,"recording_scan.avi")

            self.exposure = 25e-3
            self.fps = 40

            self.height, self.width = self.get_Dim()

            self.threshold = None
            self.mode = None

            self.InitSettings()

        else:
            print("Check cable and power button!\nAnd restart")
            Dcamapi.uninit()
            exit(1)

    def run(self):

        value = self.ctrl["mode"]

        if value == 1:
            _ = self.startLive()
        elif value == 2:
            _ = self.startZ()
        elif value == 3:
            _ = self.start()

        self.print_str.emit("Closed Everything, please press stop to reset")

        while self.ctrl["break"] == False:
            #Waiting until stop is pressed
            time.sleep(1)

        self.ctrl['break'] = False
        
        return 1

    def InitSettings(self):        
        print("Changing readout speed, success: {}".format(self.set_Value(DCAM_IDPROP.READOUTSPEED, DCAMPROP.READOUTSPEED.FASTEST)))
        print("pixel type B/W 16 bit:".format(self.set_Value(DCAM_IDPROP.IMAGE_PIXELTYPE, 2)))
        print("Color type BW success:".format(self.set_Value(DCAM_IDPROP.COLORTYPE, DCAMPROP.COLORTYPE.BW)))
        print("Bits per channel 16:".format(self.set_Value(DCAM_IDPROP.BITSPERCHANNEL, DCAMPROP.BITSPERCHANNEL._16)))
        print("exposure time set 25s, success: {}\nexposure {}".format(self.set_Value(DCAM_IDPROP.EXPOSURETIME, self.exposure), self.get_Value(DCAM_IDPROP.EXPOSURETIME)))
        print("Setting framerate to 40fps, success: {}\nfps {}".format(self.set_Value(DCAM_IDPROP.INTERNALFRAMERATE, self.fps), self.get_Value(DCAM_IDPROP.INTERNALFRAMERATE)))
    
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

        if ret == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            return 0
        
        _ = self.liveImage()
        self.dcam.buf_release()
        
        return 1

    def startZ(self):

        self.mode = 1
        self.threshold = 1000
        worked = self.allocateBuffer(self.threshold)

        if worked == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            return 0
            
        self.getFrame()
        self.dcam.buf_release()

        return 1

    def start(self):
        
        self.mode = 2
        self.threshold = 1000#3225 Change this
        worked = self.allocateBuffer(self.threshold)

        if worked == False:
            self.print_str.emit("cannot allocate memory, delete stuff or contact Arttu :)")
            return 0

        self.getFrame()
        self.dcam.buf_release()

        return 1
    
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

    def getFrame(self):
        
        self.i = 0 #num_recovered images
        self.cameraStatus = True    
        self.dcam.cap_start(bSequence=False)        
           
        if self.mode == 2:
            self.print_str.emit("Aqcuiring {} frames".format( self.threshold))
            f = open(os.path.join(self.path,'measurement_{}.csv'.format(datetime.date.today())),'w')
            f.write('index, mSec, uSec\n')
        elif self.mode == 1:
            self.print_str.emit("Aqcuiring {} frames".format(self.threshold))
       
        while self.cameraStatus:
            status = self.eventChecker()

            if (self.ctrl['break'] == True) | (self.i == self.threshold):
                #Stop if break command or threshold is reacjed
                self.dcam.cap_stop()
                self.cameraStatus = False

            elif (status==1) & (self.i < self.threshold): 
                packet = self.dcam.buf_getframe(self.i,self.color)
                
                if type(packet) == bool:
                    #if fails to revocer buffer stop
                    self.dcam.cap_stop()
                    self.cameraStatus = False
    
                iWindowStatus = self.displayframe(packet[1])
                self.i += 1
                
                if self.mode == 2:
                    f.write('{},{},{}\n'.format(self.i ,packet[0].timestamp.sec, packet[0].timestamp.microsec))

            elif status==2:

                # camera stopped
                if (self.ctrl['break'] == True) | (self.i == self.threshold):

                    self.cameraStatus = False
                    self.dcam.cap_stop()
                    
            else:
                #print("No events, {}, Something fishy going on {}".format(status,self.dcam.lasterr())) 

                self.dcam.cap_stop()
                self.cameraStatus = False
                

        if self.mode == 2:
            f.close()

        print("Camera done, recovered images {}!".format(self.i))
        return 1
        
    def close(self):
        self.print_str.emit("closing Camera")
        self.dcam.cap_stop()
        self.dcam.buf_release()
        
        self.dcam.dev_close()
        Dcamapi.uninit()


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

    
    def camera_saving(self, event_saver, q, path):
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
                packet = (packet/ 255).astype(np.uint8)
                frame = np.stack((packet,packet,packet), axis = -1)
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
    """

