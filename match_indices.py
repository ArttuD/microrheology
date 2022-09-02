# %%
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="""Match indices. This script creates new track.json files with matched indices.
                    If no automatic assignment possible, it opens a window. Click corresponding images left and right
                    in order.""")
parser.add_argument("--path",'-p', help="path to data folder. eg. G:/maria/10_10",required=True)
parser.add_argument("--radius",'-rad', help="default radius for distance calculation. Only affect cost matrix e.g 61.0",required=True,type=float)
args = parser.parse_args()

default_rad = args.radius

def op(d):
    with open(d,'r') as f:
        e = json.load(f)
    return e

coords = []
paths = glob('{}'.format(args.path))
paths = [i for i in paths if 'results' not in i]
prev = None
#default_rad = 61.0
pix_to_mm = 0.2653846153846154
all_data = []
sub = []
# find groups (repeats)
for path in paths:
    folder_name = os.path.split(path)[1]
    whole = folder_name.split('_')[0]
    w = whole[:12]
    print(w)
    #name = int(whole[12:])
    if prev is None:
        prev = w
    # if same loc sample etc...
    if prev == w:
        sub.append(path)
    else:
        all_data.append(sub)
        sub = []
        sub.append(path)
    prev = w

cols = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
for repeats in tqdm(all_data):
    # match against the first repeat
    a = os.path.join(repeats[0],'track.json')
    img1 = cv2.imread(os.path.join(repeats[0],'tracker_reference.png'))
    if img1 is None:
        print("\ntracker_reference.png not found in  {}. Creating empty img".format(repeats[0]))
        img1 = np.zeros((2048,1536,3),dtype=np.uint8)
    a_data = op(a)
    a_d = np.array([(a_data[i]['x'][0],a_data[i]['y'][0],a_data[i]['radius'][0]) for i in a_data.keys() if 'big_' in i])
    comb = np.zeros((img1.shape[0],img1.shape[1]*len(repeats),3),dtype=np.uint8)
    delta = img1.shape[1]
    comb[:,:delta,:] = img1
    unique_id = a_d.shape[0]
    #fig_1,ax_1 = plt.subplots(1,1)
    for r in range(1,len(repeats)):
        b = os.path.join(repeats[r],'track.json')

        b_data = op(b)

        b_d_ = []
        for i in b_data.keys():
            if 'big_' in i:
                rad = b_data[i]['radius']
                if len(rad) == 0:
                    rad = default_rad
                else:
                    rad = rad[0]
                b_d_.append((b_data[i]['x'][0],b_data[i]['y'][0],rad))

        b_d = np.array(b_d_)
        img2 = cv2.imread(os.path.join(repeats[r],'tracker_reference.png'))
        if img2 is None:
            img2 = np.zeros((img1.shape[0],img1.shape[1],3),dtype=np.uint8)
        comb[:,(delta*r):(delta*(r+1)),:] = img2
        # distance matrix with masking
        # x y dist
        dist_euc = np.linalg.norm(a_d[:,None,:2] - b_d[None,:,:2], axis=-1)
        dist_cond = dist_euc>(20/pix_to_mm)
        dist_euc[dist_cond] = np.inf
        dist_radius = np.linalg.norm(a_d[:,None,-1:] - b_d[None,:,-1:], axis=-1)
        dist_radius[dist_radius>(4/pix_to_mm)] = np.inf
        dist = dist_euc+dist_radius
        # radius
        #dist = np.linalg.norm(a_d[:,None,:] - b_d[None,...], axis=-1)
        # optimal assignment
        try:
            row_ind, col_ind = linear_sum_assignment(dist)
        except ValueError:
            print("no assignment: {} repeat: {}".format(repeats[0],r))
            coords = []
            fig,ax = plt.subplots(1,1)
            comb_ = np.zeros((img1.shape[0],img1.shape[1]*2,3),dtype=np.uint8)
            comb_[:,:delta,:] = img1
            comb_[:,delta:,:] = img2
            def onclick(event):
                global ix, iy
                ix, iy = event.xdata, event.ydata
                print('x = %d, y = %d'%(ix, iy))

                global coords
                coords.append((ix, iy))

                return coords
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            fig.show()
            while True:
                ax.imshow(comb_)
                if plt.waitforbuttonpress():
                    plt.close()
                    break
                plt.pause(0.01)
            fig.canvas.mpl_disconnect(cid)
            col_ind = [-1]*(len(coords)//2)
            for i in range(0,len(coords),2):
                # find closest match
                #print(np.linalg.norm(coords[i]-a_d[:,:2],axis=1))
                closest_a = np.argmin(np.linalg.norm(coords[i]-a_d[:,:2],axis=1))
                coord_right = coords[i+1]
                coord_right = (coord_right[0]-delta,coord_right[1])
                closest_b = np.argmin(np.linalg.norm(coord_right-b_d[:,:2],axis=1))
                col_ind[closest_b] = closest_a
            c_i = np.array(col_ind)
            row_ind = np.arange(c_i.shape[0])[c_i != -1]
            print(row_ind)
            col_ind = c_i[c_i != -1]
            print(col_ind)
        #fig = plt.figure(figsize=(20,20))
        

        # mapping vis
        matched_data = {}
        cc = cols[r%len(cols)]
        for idx,ind in zip(row_ind,col_ind):
            # b (col index) mapped to a (row index)
            proposal = b_data['big_{}'.format(ind)]
            x_proposal = int(proposal['x'][0])+delta*r
            y_proposal = int(proposal['y'][0])
            new_id = 'big_{}'.format(idx)
            matched_data[new_id] = {}
            matched_data[new_id]['x'] = proposal['x']
            matched_data[new_id]['y'] = proposal['y']
            matched_data[new_id]['radius'] = proposal['radius']
            matched_data[new_id]['label'] = proposal['label']
            matched_data[new_id]['timestamps'] = proposal['timestamps']

            origin = a_data['big_{}'.format(idx)]
            x_origin = int(origin['x'][0])
            y_origin = int(origin['y'][0])

            comb = cv2.line(comb, (x_proposal,y_proposal), (x_origin,y_origin), cc, 5)
        
        # add unmatched
        indices = np.arange(b_d.shape[0])
        for i in indices:
            if i not in col_ind:
                print('adding unique id {} in {} repeat: {}'.format(unique_id,os.path.split(repeats[0])[-1],r))
                new_id = 'big_{}'.format(unique_id)
                matched_data[new_id] = {}
                matched_data[new_id]['x'] = proposal['x']
                matched_data[new_id]['y'] = proposal['y']
                matched_data[new_id]['radius'] = proposal['radius']
                matched_data[new_id]['label'] = proposal['label']
                matched_data[new_id]['timestamps'] = proposal['timestamps']
                unique_id += 1

        for key in b_data.keys():
            if 'ref_' in key:
                new_id = key
                matched_data[new_id] = {}
                matched_data[new_id]['x'] = b_data[new_id]['x']
                matched_data[new_id]['y'] = b_data[new_id]['y']
                matched_data[new_id]['radius'] = b_data[new_id]['radius']
                matched_data[new_id]['label'] = b_data[new_id]['label']
                matched_data[new_id]['timestamps'] = b_data[new_id]['timestamps']

        with open(os.path.join(repeats[r],'track_matched.json'),'w') as f:
            json.dump(matched_data,f)

    cv2.imwrite(os.path.join(repeats[0],'track_reference_matched.png'),comb)
    #ax_1.imshow(comb)
# %%
