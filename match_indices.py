# %%
import json
from PIL.Image import new
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="""Match indices. This script creates new track.json files with matched indices.
                    If no automatic assignment possible, it opens a window. Click corresponding images left and right
                    in order.""")
parser.add_argument("--path",'-p', help="path to data folder. eg. G:/maria/10_10",required=True)
parser.add_argument("--radius",'-rad', help="maximum distance between two samples in cluster",default=8.0,type=float)
parser.add_argument('--vis','-v',help='Visualize match',
                    action="store_true")
args = parser.parse_args()

def op(d):
    with open(d,'r') as f:
        e = json.load(f)
    return e

coords = []
paths = glob('{}'.format(args.path))
paths = [i for i in paths if 'results' not in i]
prev = None
#default_rad = 61.0
#pix_to_um = 3.45/(20)
pix_to_um = 3.45/(20*.63)
default_rad = args.radius
all_data = []
sub = []
# find groups (repeats)
for path in paths:
    folder_name = os.path.split(path)[1]
    whole = folder_name.split('_')[0]
    w = whole[:12]
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
if len(sub) > 1:
    all_data.append(sub)

cols = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
for idx,repeats in tqdm(enumerate(all_data)):
    img1 = np.zeros((1544,2064,3),dtype=np.uint8)
    comb = np.zeros((img1.shape[0],img1.shape[1]*len(repeats),3),dtype=np.uint8)
    delta = img1.shape[1]
    print(repeats[0])
    track_data = []
    info = []
    repeat_id = []
    key_names = []
    for r in range(len(repeats)):
        b = os.path.join(repeats[r],'track.json')
        #print(os.path.join(repeats[r],'tracker_reference.png'))
        img2 = cv2.imread(os.path.join(repeats[r],'tracker_reference.png'))
        comb[:,(delta*r):(delta*(r+1)),:] = img2
        b_data = op(b)
        track_data.append(b_data)
        b_keys = []
        b_d_ = []
        for i in b_data.keys():
            if 'big_' in i:
                b_keys.append(i)
                rad = b_data[i]['radius']
                if len(rad) == 0:
                    rad = default_rad
                else:
                    rad = rad[0]
                b_d_.append((b_data[i]['x'][0],b_data[i]['y'][0],rad))

        b_d = np.array(b_d_)
        info.append(b_d)
        repeat_id.append([r]*b_d.shape[0])
        key_names.append(b_keys)
    xy = np.concatenate([i[:,:2] for i in info])
    repeat_id = np.concatenate(repeat_id)
    key_names = np.concatenate(key_names)
    clustering = DBSCAN(eps=2*default_rad, min_samples=1).fit(xy*pix_to_um)
    for idx2,i in enumerate(np.unique(clustering.labels_)):
        c = clustering.labels_==i
        x = repeat_id[c]*img1.shape[1]+xy[c,0]
        y = xy[c,1]
        pts = np.stack([x,y]).T.reshape((-1,1,2)).astype(int)
        comb = cv2.polylines(comb, [pts], False, (0,255,0), 5)
        for k,j in zip(repeat_id[c],xy[c,:2]):
            comb = cv2.drawMarker(comb, (int(j[0]+k*img1.shape[1]),int(j[1])), (255, 0, 0), 0, 30, 4)
            #populate track matched
        #print(repeat_id[c])
        #for track_id,k_name in zip(repeat_id[c],key_names[c]):
        #    #print(f'repeat: {track_id} new: big_{i} pop: {k_name}')
        #    track_data[track_id]['big_{}'.format(idx2)] = track_data[track_id].pop(k_name)
        plt.scatter(xy[c,0],xy[c,1])
        plt.text(np.mean(xy[c,0]),np.mean(xy[c,1]),f'{idx2}')
    #print(repeat_id,clustering.labels_,key_names)
    track_data_2 = []
    for i in range(len(track_data)):
        track_data_2.append({})
    for i in range(len(track_data)):
        for j in list(track_data[i].keys()):
            if 'big_' not in j:
                track_data_2[i][j] = track_data[i][j]
    for track_id,new_name,old_names in zip(repeat_id,clustering.labels_,key_names):
        #print(f'repeat: {track_id} new: big_{i} pop: {k_name}')
        track_data_2[track_id]['big_{}'.format(new_name)] = track_data[track_id].pop(old_names)
    for r in range(len(repeats)):
        b = os.path.join(repeats[r],'track_matched.json')
        with open(b,'w') as f:
            json.dump(track_data_2[r],f)
    for i in track_data_2:
        for j in list(i.keys()):
            if 'big_' in j:
                plt.text(i[j]['x'][0],i[j]['y'][0],'{}'.format(j))
    cv2.imwrite(os.path.join(repeats[0],'track_reference_matched.png'),comb)
    if args.vis:
        plt.show()
    else:
        plt.close()