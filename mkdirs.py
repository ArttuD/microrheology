from glob import glob
import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="""Group data
                """)
parser.add_argument('--path','-p',required=True,
                    help='Path to folder. eg. C:/data/imgs')
parser.add_argument('--date','-d',required=True,
                    help='Date of experiments as YMD 230301')
parser.add_argument('--step','-s',required=False,
                    help='How many imgs in hour',default=4)

args = parser.parse_args()

def sort(data):
    indices = [int(os.path.split(i)[1].split('_')[1].split('.')[0]) for i in data]
    return data[np.argsort(indices)]

imgs = np.array(glob(os.path.join(args.path,'img_*')))
fluor = np.array(glob(os.path.join(args.path,'fluor_*')))
video = np.array(glob(os.path.join(args.path,'video_*')))

imgs = sort(imgs)
fluor = sort(fluor)
video = sort(video)

def move(data,idx,step,out):
    files = data[(idx*step):((idx+1)*step)]
    if len(files)>0:
        for i in files:
            out_name = os.path.join(out,os.path.split(i)[-1])
            print(f'{i} -> {out_name}')
            os.rename(i,out_name)


path = os.path.join(args.path,args.date)
if not os.path.exists(path):
    print(path)
    os.mkdir(path)

# create folders 
n = len(video)//2
for i in range(n):
    f_path = os.path.join(path,'{}0101{:02d}01_t'.format(args.date,i+1))
    if not os.path.exists(f_path):
        print(f'--- {f_path}')
        os.mkdir(f_path)
    # move stuff to folder
    move(imgs,i,args.step,f_path)
    move(fluor,i,args.step,f_path)
    move(video,i,2,f_path)


    
trials = pd.read_csv(os.path.join(args.path,'datalogger.csv'),sep='\t',header=None)
diffs = trials.iloc[:,-1].diff()
diffs[pd.isna(diffs)] = diffs.median()
trials['group'] = (diffs.abs()>10).cumsum()
print('trial')
for i,(index,trial) in enumerate(trials.groupby('group')):
    f_path = os.path.join(path,'{}0101{:02d}01_t'.format(args.date,i+1),'23_trial_.csv')
    if os.path.exists(f_path):
        print(f_path)
        trial.iloc[:,[2,3,0,0,0,0,4,5,6]].to_csv(f_path,sep='\t',header=False,index=False)


