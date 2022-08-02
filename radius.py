# %%
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
import cv2
from tqdm import tqdm
import pandas as pd
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from tools.radius_estimator import RadiusEstimator

# %%

#Ask path of data
parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)

parser.add_argument('--path','-p',required=True,
                    help='Path to folder. eg. C:/data/imgs')
parser.add_argument("--visualize",'-v', help="Visualize tracks",
                    action="store_true", default=True)
parser.add_argument("--init",'-i', help="use existing track.json info",
                    action="store_true", default=True)
parser.add_argument('--copy','-c',help='Copy radius estimation info to repeats',
                    action="store_true",default=True)


#Save arguments
args = parser.parse_args()
path = args.path
args_dict = vars(args)
estimator = RadiusEstimator()

#Find z-stack video from the folders of each first repeat (if does not exist takes actual measurement)
for path_R in glob(os.path.join(args.path,'*01_*')):
    if '1_' in path_R:
        print("path above")  
        vids = glob('{}/*.mp4'.format(path_R))
        imgs = vids[np.argmin([os.path.getsize(i) for i in vids ])]
        args_dict['path'] = imgs
        paths = None
        if args.copy:
            paths = glob('{}*'.format(path_R.split('_')[0][:-2]))
    #run radius estimator
    estimator.process_folder(args_dict,paths)


