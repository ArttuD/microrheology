# %%
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.saver import Saver
import time
import datetime
import argparse
import os
import json
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq
import sys
# %%


class ChangeFocus:
    def __init__(self,imgs,rad_plots,focus_plots,focus_lines,rad_lines,img_plots,all_rads,out_path,rads):
        
        self.imgs = imgs
        self.focus_plots = focus_plots
        self.rad_plots = rad_plots
        self.img_plots = img_plots
        self.focus_lines = focus_lines
        self.rad_lines = rad_lines
        self.t = [0]*len(imgs)
        self.all_rads = all_rads
        self.changed = False
        self.cid = self.focus_lines[0].figure.canvas.mpl_connect('button_press_event', self)
        self.close = self.focus_lines[0].figure.canvas.mpl_connect('close_event', self.on_close)
        self.out_path = out_path
        self.rads = rads
        self.map = {idx:i for idx,i in enumerate(rads.keys())}

    @staticmethod
    def find(axes,ax):
        for idx,i in enumerate(axes):
            if i == ax:
                return idx
        return 0

    def load_img(self,idx,a_id):
        img = self.imgs[a_id][idx]
        self.img_plots[a_id].set_array(img)

    def on_close(self,event):
        self.focus_lines[0].figure.savefig('{}'.format(os.path.join(self.out_path,'radius_estimate.png')))

    def get_rads(self):
        return self.rads

    def __call__(self, event):
        x = event.xdata
        y = event.ydata
        if event.inaxes in self.focus_plots:
            a_id = self.find(self.focus_plots,event.inaxes)
            x_round = int(np.round(x))
            self.focus_lines[a_id].set_data([x,x],[0,1])
            self.load_img(x_round,a_id)
            c = self.all_rads[a_id][x_round]
            self.rad_lines[a_id].set_data([0,1],[c,c])
            self.focus_lines[a_id].figure.canvas.draw()
            self.t[a_id] = x_round
            self.changed = True

            self.rads[self.map[a_id]] = c

        elif event.inaxes in self.rad_plots:
            a_id = self.find(self.rad_plots,event.inaxes)
            x_round = int(np.round(x))
            c = self.all_rads[a_id][x_round]
            self.rad_lines[a_id].set_data([0,1],[c,c])
            self.load_img(x_round,a_id)
            x_round = int(np.round(x))
            self.focus_lines[a_id].set_data([x,x],[0,1])
            self.focus_lines[a_id].figure.canvas.draw()
            self.t[a_id] = x_round
            self.changed = True

            self.rads[self.map[a_id]] = c

    def disconnect(self):
        self.focus_lines[0].figure.canvas.mpl_disconnect(self.cid)
        self.focus_lines[0].figure.canvas.mpl_disconnect(self.close)



class RadiusEstimator():
    
    def __init__(self) -> None:

        # fun colors
        self.colors_ref = [(114,229,239), (139,18,58), (143,202,64), 
        (118,42,172), (81,243,16), (235,107,230), (35,94,49),
        (197,213,240)]
        self.colors_big = [(62,71,86), (13,243,143), (241,67,48),
        (11,164,126), (250,33,127), (215,199,123),
        (27,77,171), (250,175,227)]
        self.m = 3.45/(20)

        self.image = None
        cv2.namedWindow('win',cv2.WINDOW_NORMAL)
        #self.refPt = []
        #self.final_boundaries = []

    def click_and_crop(self,event, x, y, flags, param):
        # mouse events
        # get x,y,x2,y2 coords
        #global self.refPt, image
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            self.final_boundaries.append((self.refPt[0],self.refPt[1]))
            cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 4)
            cv2.imshow("win", self.image)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            clone = self.image.copy()
            cv2.rectangle(clone, self.refPt[0], (x, y), (0, 255, 0), 4)
            cv2.imshow("win", clone)

    def click_and_draw(self,event, x, y, flags, param):
        # mouse events
        # get x,y,x2,y2 coords
        #global self.refPt, image
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = [(x, y)]
            self.draw = np.copy(self.draw_start)
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x, y))
            rad = np.sqrt((self.refPt[0][0]-x)**2+(self.refPt[0][1]-y)**2)
            self.final_boundaries.append((self.refPt[0],rad))
            cv2.circle(self.draw, self.refPt[1], int(rad), (0, 255, 0), 1)
            cv2.imshow("drawer", self.draw)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            clone = self.draw.copy()
            rad = int(np.sqrt((self.refPt[0][0]-x)**2+(self.refPt[0][1]-y)**2))
            cv2.circle(clone, (x,y), rad, (0, 255, 0), 1)
            cv2.imshow("drawer", clone)


    def get_track_info(self,path):
        coords = {}
        track_path = os.path.join(path,'track_matched.json')
        choose = 0
        if os.path.exists(track_path):
            with open(track_path,'r') as f:
                track_info = json.load(f)
            for i in track_info.keys():
                if 'big' in i:
                    x_ = track_info[i]['x'][0]
                    y_ = track_info[i]['y'][0]
                    shift = np.median(track_info[i]['radius'])+10
                    #shift = 30
                    coords[i] = {'y':x_-shift,'y2':x_+shift,'x':y_-shift,'x2':y_+shift}
            choose = 1
        return coords,choose

    def process_folder(self,args,paths):
        """Process data inside folder. Parser is dict of values"""

        #Save path
        path = args['path']
        vis = not args['visualize']

        #Save all the images in the folder
        imgs = path

        # needed for the saver
        json_name = '%s/track_matched.json'%path

        out_path = path
        if out_path.endswith('.mp4'):
            out_path = os.path.split(out_path)[0]

        print("Saving data to: {}".format(out_path))

        self.refPt = []
        self.final_boundaries = []
        self.image = None
        #Saving variable
        saver = Saver(json_name)

        #Coordinates
        coords = {}

        choose = 0
        # if using track_matched.json xy info
        if not args['init']:
            if not args['repeats']:
                coords,choose = self.get_track_info(os.path.split(path)[0])
            else:
                # if we have repeats, find all probes. e.g repeat 2 might
                # have probes which are not tracked in repeat 1...
                # might cause problems if large drifts over repeats
                cs = []
                for i in paths:
                    coords_,choose = self.get_track_info(i)
                    cs.append(coords_)

                # all from first repeat
                coords = cs[0]
                # add keys from the rest of the repeats
                for i in range(1,len(cs)):
                    a = list(cs[i-1].keys())
                    b = list(filter(lambda x: x not in a,list(cs[i].keys())))
                    for k in b:
                        coords[k] = cs[i][k]

        #Create a window
        image = None
        cv2.namedWindow('win',cv2.WINDOW_NORMAL)

        cv2.setMouseCallback("win", self.click_and_crop)

        i = 0

        first = True
        self.first_img = None

        #Import and process images, download bar
        for k1 in range(2):
            cap = cv2.VideoCapture(imgs)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm(total=length)

            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
            calc_timestamps = [0.0]

            while cap.isOpened():
                #Download image, save the shape, and define color
                frame_exists, img = cap.read()
                if frame_exists:
                    timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                    calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
                else:
                    break
                if img is None:
                    break
                (w,h,_) = img.shape
                image_out = np.copy(img)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if first:
                    first = False
                    self.first_img = np.copy(image_out)
                
                # choose boxes
                if choose==0:
                    self.image = np.copy(img)
                    cv2.imshow('win',self.image)
                    # big guys
                    k = cv2.waitKey(0) & 0xFF
                    for idx,b in enumerate(self.final_boundaries):
                        coords['big_%i'%idx] = {'x':b[0][1],'x2':b[1][1],'y':b[0][0],'y2':b[1][0]}
                    self.final_boundaries = []
                    choose += 1
                
                # Operate in small windows, loop over particles to track
                for key in coords.keys():
                    #Coordinates save
                    coord = coords[key]
                    x = int(np.floor(coord['x']))
                    x2 = int(np.floor(coord['x2']))
                    y = int(np.floor(coord['y']))
                    y2 = int(np.floor(coord['y2']))
                    # take crop
                    sub_img = img[x:x2,y:y2]

                    # find binarization
                    th = None
                    if k1==1:
                        th = np.percentile(saver.tracks[key]['threshold'],5)
                        ret3,binary = cv2.threshold(sub_img,th,255,cv2.THRESH_BINARY)
                    else:
                        ret3,binary = cv2.threshold(sub_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    if binary is None:
                        binary = np.zeros_like(sub_img)
                    else:
                        binary = 255-binary


                    split = key.split('_')
                    
                    # find largest connected area
                    contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    radius = -1
                    if contours:
                        biggest = max(contours, key = cv2.contourArea)
                        (x_,y_),radius = cv2.minEnclosingCircle(biggest)
                        center = (int(x_)+int(np.round(coords[key]['y'])),int(y_)+int(np.round(coords[key]['x'])))
                        c = int(split[-1])
                        col = None
                        if split[0] == 'ref':
                            col = self.colors_ref[(c+1)%8]
                        else:
                            col = self.colors_big[(c+1)%8]
                        image_out[x:x2,y:y2,:] = cv2.addWeighted(image_out[x:x2,y:y2,:],1.0,np.tile(binary[...,np.newaxis],(1,3)),0.5,0.0)
                        image_out = cv2.circle(image_out,center,int(radius),col,4)
                    else:
                        biggest = np.zeros_like(binary)


                    # old center
                    old_x = (coord['x2']-coord['x'])/2
                    old_y = (coord['y2']-coord['y'])/2

                    # find center
                    M = cv2.moments(biggest)
                    if M["m00"]==0:
                        cX = old_x
                        cY = old_y
                    else:
                        cY = M["m10"] / M["m00"]
                        cX = M["m01"] / M["m00"]

                    start_pos = (int(np.round(coords[key]['y'])),int(np.round(coords[key]['x'])))
                    end_pos = (int(np.round(coords[key]['y2'])),int(np.round(coords[key]['x2'])))
                    # pretty colors
                    c = int(split[-1])
                    col = None
                    if split[0] == 'ref':
                        col = self.colors_ref[(c+1)%8]
                    else:
                        col = self.colors_big[(c+1)%8]
                    # visualize
                    if vis:
                        cv2.rectangle(image_out, start_pos, end_pos, col, 5)
                    #add labels var 
                    if (sub_img.shape[0] == 0) or (sub_img.shape[1] == 0):
                        lap = np.inf
                    else:
                        lap = cv2.Laplacian(sub_img, cv2.CV_64F).var()
                    if vis:
                        cv2.putText(image_out,'{}: th: {} lap: {}'.format(key, th, lap),(start_pos[0],start_pos[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,col,3)
                    # mid point to save
                    mid_y = (coords[key]['y2']+coords[key]['y'])/2
                    mid_x = (coords[key]['x2']+coords[key]['x'])/2
                    label = 0
                    if split[0] == 'big':
                        label = 1
                    # notice that coordinates are flipped
                    saver.add_key('threshold')
                    saver.add_key('laplace')
                    saver.update({'label':label,'id':key,'x':mid_y,'y':mid_x,'timestamp':i, 'radius': radius},['threshold','laplace'],[ret3,lap])

                #Show image and break if press ctrl+C
                if k1 == 0:
                    if vis:
                        cv2.putText(image_out,'Estimating thresholds',(0,80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)
                else:
                    if vis:
                        cv2.putText(image_out,'Computing radius',(0,80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3)
                if vis:
                    cv2.imshow('win',image_out)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
                i += 1
                pbar.update(1)

        cap.release()
        cv2.destroyAllWindows()

        fig,ax = plt.subplots(3,len(saver.tracks.keys()))

        if len(saver.tracks.keys()) == 1:
            ax = np.expand_dims(ax,1)
        rad_lines = []
        focus_lines = []
        max_indices = []
        out_dict = {}
        rad_names = []
        rad_maxes = []
        all_rads = []
        for idx,i in enumerate(saver.tracks.keys()):
            rad = np.array(saver.tracks[i]['radius'])
            rad = rad[rad.shape[0]//2:]
            laplace = np.array(saver.tracks[i]['laplace'])
            laplace = gaussian_filter1d(laplace[laplace.shape[0]//2:],10)
            #np.save('G:/rad_{}.npy'.format(idx),rad)
            ax[0,idx].plot(rad)
            ax[1,idx].plot(laplace)
            laplace[np.isnan(laplace)] = -np.inf
            lap_max = np.argmax(laplace)
            max_indices.append(lap_max)
            if lap_max >= len(rad):
                lap_max = len(rad)-1
            if len(rad) == 0:
                print(f'No radius for: {i}. Quitting')
                sys.exit(0)
            else:
                rad_max = rad[lap_max]
            rad_maxes.append(rad_max)
            rad_names.append(i)
            out_dict[i] = rad_max
            all_rads.append(rad)
            l1 = ax[0,idx].axhline(rad_max,color='red')
            l2 = ax[1,idx].axvline(lap_max,color='red')
            rad_lines.append(l1)
            focus_lines.append(l2)


        max_indices_sort = np.sort(max_indices)
        ind_sorted = np.argsort(max_indices)
        cur_frame = 0
        cur_ind = 0
        ind = ind_sorted[cur_ind]
        cap = cv2.VideoCapture(imgs)
        img_plots = [None]*len(list(saver.tracks.keys()))
        while cap.isOpened():
            #Download image, save the shape, and define color
            frame_exists, img = cap.read()
            if not frame_exists:
                break
            if cur_ind == max_indices_sort.shape[0]:
                break
            # find all particles that have best focus
            while cur_frame == max_indices_sort[cur_ind]:
                rad_val = rad_maxes[ind]+30
                y = max(int(saver.tracks[list(saver.tracks.keys())[ind]]['x'][0]-rad_val),0)
                x = max(int(saver.tracks[list(saver.tracks.keys())[ind]]['y'][0]-rad_val),0)
                x_end = min(int(x+rad_val*2),img.shape[0])
                y_end = min(int(y+rad_val*2),img.shape[1])
                sub_img = img[x:x_end,y:y_end,:]
                img_plots[ind] = ax[2,ind].imshow(sub_img)
                cur_ind += 1
                if cur_ind==ind_sorted.shape[0]:
                    break
                ind = ind_sorted[cur_ind]
            cur_frame += 1

        cap.release()
        all_imgs = []
        ll = len(list(saver.tracks.keys()))
        for i in range(ll):
            all_imgs.append([])
        cap = cv2.VideoCapture(imgs)
        while cap.isOpened():
            frame_exists, img = cap.read()
            if not frame_exists:
                break
            for i in range(ll):
                rad_val = rad_maxes[i]#+30
                y = max(int(saver.tracks[list(saver.tracks.keys())[i]]['x'][0]-rad_val),0)
                x = max(int(saver.tracks[list(saver.tracks.keys())[i]]['y'][0]-rad_val),0)
                x_end = min(int(x+rad_val*2),img.shape[0])
                y_end = min(int(y+rad_val*2),img.shape[1])
                sub_img = img[x:x_end,y:y_end,:]
                all_imgs[i].append(sub_img)
        cap.release()

        ax[0,0].set_title("Radius (pixels)")
        ax[1,0].set_title("Blur (higher = better focus)")
        self.handler = ChangeFocus(all_imgs,ax[0],ax[1],focus_lines,rad_lines,img_plots,all_rads,out_path,out_dict)
        plt.show()
        out_dict = self.handler.get_rads()
        self.handler.disconnect()
        response = input('Continue? (if no write n)')
        if response == 'n':
            if len(saver.tracks.keys()) != 1:
                response2 = input('which ones= (, separated e.g 0,2,5): ')
                response_values = [int(i) for i in response2.split(',')]
            else:
                response_values = [0]
            for i in response_values:
                self.refPt = []
                self.final_boundaries = []
                cv2.namedWindow('drawer',cv2.WINDOW_NORMAL)
                cv2.setMouseCallback("drawer", self.click_and_draw)
                drawing = True
                rad_val = rad_maxes[i]+30
                y = max(int(saver.tracks[list(saver.tracks.keys())[i]]['x'][0]-rad_val),0)
                x = max(int(saver.tracks[list(saver.tracks.keys())[i]]['y'][0]-rad_val),0)
                x_end = min(int(x+rad_val*2),self.first_img.shape[0])
                y_end = min(int(y+rad_val*2),self.first_img.shape[1])
                self.draw_start = np.copy(all_imgs[i][self.handler.t[i]])
                self.draw = np.copy(self.draw_start)
                cv2.imshow('drawer',self.draw)
                while drawing:

                    if cv2.waitKey(1) & 0xFF == 27:
                        drawing = False

                out_dict['big_{}'.format(i)] = self.final_boundaries[-1][1]

                cv2.destroyAllWindows()

        print(out_path)
        del self.handler
        # save results
        with open('{}'.format(os.path.join(out_path,'radius_estimates.json')),'w') as f:
            json.dump(out_dict,f)
        
        #print(paths)
        if paths != None:
            for p in paths[1:]:
                print('copying to: {}'.format(os.path.split(p)[1]))
                #print('copying to: {}'.format(os.path.join(p,'radius_estimates.json')))
                
                with open('{}'.format(os.path.join(p,'radius_estimates.json')),'w') as f:
                    json.dump(out_dict,f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Estimate radius with more accuracy from sweep videos""")

    parser.add_argument('--path','-p',required=True,
                        help='Path to folder. eg. C:/data/imgs')
    parser.add_argument("--no_visualize",'-v', help="Don't visualize tracks",
                        action="store_true")
    parser.add_argument("--no_init",'-i', help="Do not use existing track.json info",
                        action="store_true")
    parser.add_argument('--no_copy','-c',help='Do not copy radius estimation info to repeats',
                        action="store_true")
    parser.add_argument('--repeats',help='Run estimation for all repeats',
                        action="store_true")

    args = parser.parse_args()
    args_dict = vars(args)
    
    estimator = RadiusEstimator()
    
    for path in glob(os.path.join(args.path,'*')): 
        if '01_' in path:    
            vids = glob('{}/*.mp4'.format(path))
            imgs = vids[np.argmin([ os.path.getsize(i) for i in vids ])]
            args_dict['path'] = imgs
            paths = None
            if not args.copy:
                paths = glob('{}*'.format(path.split('_')[0][:-2]))
            estimator.process_folder(args_dict,paths)
                