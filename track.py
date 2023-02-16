from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from tools.saver import Saver
import argparse
import os
import sys

#Incease birghtness if the video is dark (visualization)
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#Colors
colors_ref = [(114,229,239), (139,18,58), (143,202,64), 
(118,42,172), (81,243,16), (235,107,230), (35,94,49),
(197,213,240)]
colors_big = [(62,71,86), (13,243,143), (241,67,48),
(11,164,126), (250,33,127), (215,199,123),
(27,77,171), (250,175,227)]

#Ask path of data
parser = argparse.ArgumentParser(
    description="""Semi automatic tracking for magnetic particles.
                Double right click to remove tracks.
                """)
parser.add_argument('--path','-p',required=False,
                    help='Path to folder. eg. C:/data/imgs')
parser.add_argument("--visualize",'-v', help="Display video while tracking",
                    action="store_true",default = True)
parser.add_argument("--cells",'-c', help="Label cell centers",
                    action="store_true",default = True)
parser.add_argument('--brightness','-b',required=False,
                    help='Optional brigtness increase. e.g 20 (integer)',type=int,default=0)
parser.add_argument("--filter",'-f', help="Remove white reflections from particles",
                    action="store_true",required=False)
#Save arguments
args = parser.parse_args()

path = args.path
vis = args.visualize
brightness = args.brightness
track_cells = args.cells

#Pick acutal measurement video from each folder
vids = glob('{}/*.mp4'.format(path))
if len(vids)==0:
    vids = glob('{}/*.avi'.format(path))
    if len(vids)==0:
        print(f'No videos in {path}. Exiting')
        sys.exit(0)
imgs = vids[np.argmax([ os.path.getsize(i) for i in vids ])]

#create a file where data is saved
json_name = '%s/track.json'%path
print("Save path: %s"%json_name)

#Saving variable
saver = Saver(json_name)

#Initiate
coords = {}
refPt = []
final_boundaries = []
to_remove = []

#Create a window
image = None
cv2.namedWindow('win',cv2.WINDOW_NORMAL)

# mouse events
def click_events(event, x, y, flags, param):
    #Get x,y and draw boxes
    global refPt, image, to_remove
    if event == cv2.EVENT_RBUTTONDBLCLK:
        to_remove = [(x,y)]
    elif event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        final_boundaries.append((refPt[0],refPt[1])) 
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 4)
        cv2.imshow("win", image)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        clone = image.copy()
        cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 4)
        cv2.imshow("win", clone)


def click_remove(event, x, y, flags, param):
    #remove lost tracks
    global refPt, image
    if event == cv2.EVENT_RBUTTONDBLCLK:
        print("to remove")

cv2.setMouseCallback("win", click_events)

#initiate more parametes
i = 0
choose = 0
keys_dropped = []

#Import and process images, download bar
print(imgs)
cap = cv2.VideoCapture(imgs)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=length)

#Read fetatures
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]

while cap.isOpened():
    #Download image, save the shape, and define color
    frame_exists, img = cap.read()
    if frame_exists:
        skip_count = 1
        if track_cells:
            skip_count += 1
        if choose>skip_count:
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
    else:
        break
    if img is None:
        break
    (w,h,_) = img.shape
    
    # preprocess the frame 
    if brightness != 0:
        img = increase_brightness(img,brightness)
    if args.filter:
        img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))
    
    image_out = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # choose boxes
    
    #Magnetic particles
    if choose==0:
        image = np.copy(image_out)
        cv2.imshow('win',image)
        k = cv2.waitKey(0) & 0xFF
        for idx,b in enumerate(final_boundaries):
            coords['big_%i'%idx] = {'x':b[0][1],'x2':b[1][1],'y':b[0][0],'y2':b[1][0]}
        final_boundaries = []
        choose += 1

    #Reference particles
    if choose==1:
        # reference guys #change
        image = np.copy(image_out)
        cv2.imshow('win',image)       
        k = cv2.waitKey(0) & 0xFF
        for idx,b in enumerate(final_boundaries):
            coords['ref_%i'%idx] = {'x':b[0][1],'x2':b[1][1],'y':b[0][0],'y2':b[1][0]}
        choose += 1
        final_boundaries = []
        if not vis:
            cv2.destroyAllWindows()

    #Cells if wanted
    
    if track_cells and choose==2:
        # reference guys
        image = np.copy(image_out)
        cv2.imshow('win',image)       
        k = cv2.waitKey(0) & 0xFF
        for idx,b in enumerate(final_boundaries):
            coords['cell_%i'%idx] = {'x':b[0][1],'x2':b[1][1],'y':b[0][0],'y2':b[1][0]}
        choose += 1
        final_boundaries = []
        if not vis:
            cv2.destroyAllWindows()
    
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
        if 'cell_' not in key:
            # find binarization
            ret3,binary = cv2.threshold(sub_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if binary is None:
                binary = np.zeros_like(sub_img)
            else:
                binary = 255-binary

            # find largest connected area
            contours,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            radius = -1
            if contours:
                biggest = max(contours, key = cv2.contourArea)
                (x_,y_),radius = cv2.minEnclosingCircle(biggest)
                center = (int(x_),int(y_))
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

            # fix int precision error
            y_int_error = y-coord['y']
            x_int_error = x-coord['x']

            # update global coordinates for this track
            shift_x = cX-old_x+x_int_error
            shift_y = cY-old_y+y_int_error
            coords[key]['x'] += shift_x
            coords[key]['x2'] += shift_x
            coords[key]['y'] += shift_y 
            coords[key]['y2'] += shift_y

            # validate that coordinates are inside image
            x_max = img.shape[0]
            y_max = img.shape[1]
            # upper limit
            if x>=x_max:
                coords[key]['x'] = x_max-1
            if x2>=x_max:
                coords[key]['x2'] = x_max-1
            if y>=y_max:
                coords[key]['y'] = y_max-1
            if y2>=y_max:
                coords[key]['y2'] = y_max-1
            # lower limit
            if x<0:
                coords[key]['x'] = 0
            if x2<0:
                coords[key]['x2'] = 0
            if y<0:
                coords[key]['y'] = 0
            if y2<0:
                coords[key]['y2'] = 0
        
        split = key.split('_')
        
        #Visualize tracking
        if vis:
            #Draw boxes
            start_pos = (int(np.round(coords[key]['y'])),int(np.round(coords[key]['x'])))
            end_pos = (int(np.round(coords[key]['y2'])),int(np.round(coords[key]['x2'])))
            c = int(split[-1])
            col = None
            if split[0] == 'ref':
                col = colors_ref[(c+1)%8]
            elif split[0] == 'cell':
                col = (0,0,255)
            else:
                col = colors_big[(c+1)%8]
            # display
            cv2.rectangle(image_out, start_pos, end_pos, col, 5)
            #add labels
            cv2.putText(image_out,'{}'.format(key),(start_pos[0],start_pos[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1,col,3)

        # mid point to save
        mid_y = (coords[key]['y2']+coords[key]['y'])/2
        mid_x = (coords[key]['x2']+coords[key]['x'])/2
        label = 0
        if split[0] == 'big':
            label = 1
        elif split[0] == 'cell':
            label = 2
        # remove track
        if len(to_remove) != 0:
            coord_coords = np.array([(coords[c]['y'],coords[c]['x']) for c in list(coords.keys())])
            if len(coord_coords.shape)==1:
                coord_coords = np.reshape(coord_coords,(1,-1))
            to_remove_np = np.array(to_remove)
            dists = np.linalg.norm(coord_coords-to_remove_np,axis=1)
            closest = np.argmin(dists)
            drop_key = list(coords.keys())[closest]
            print("removing: {}".format(drop_key))
            keys_dropped.append(drop_key)
            to_remove = []
        # notice that coordinates are flipped
        saver.update({'label':label,'id':key,'x':mid_y,'y':mid_x,'timestamp':i, 'radius': radius})

    #Show image and break if press ctrl+C
    if vis:
        cv2.imshow('win',image_out)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    i += 1
    pbar.update(1)

    # drop keys
    for keys_to_drop in keys_dropped:
        coords.pop(drop_key,None)

#Save one visualized frame
saver.save(keys_dropped)
cv2.imwrite(os.path.join(path,'tracker_reference.png'),image_out)
cap.release()
cv2.destroyAllWindows()

# save timestamp file
total_diff = np.median(np.diff(timestamps))
with open('{}/frame_info_matlab.txt'.format(path),'w') as f:
    prev = timestamps[0]
    for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
        val = 0.0
        diff = abs(ts - cts)
        if diff != 0:
            val = diff
        if ts == 0 and i>0:
            f.write('{}: {}\n'.format(i+1,(prev+total_diff)/1000.0))
            prev = prev+total_diff
            
        else:
            f.write('{}: {}\n'.format(i+1,ts/1000.0))
            prev = ts