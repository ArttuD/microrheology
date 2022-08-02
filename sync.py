# %%
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
from tools.syncronizer import syncronizer

# %%
#Ask path of data
parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)

parser.add_argument('--path','-p',required=True,
                    help='Path to folder. eg. C:/data/imgs')


#Save arguments
args = parser.parse_args()
args_dict = vars(args)
path = args.path


#Initialize parameters for syncronizer
used = False;auto_on = True; avg_preprocess = True

syncr = syncronizer(path,used,auto_on,avg_preprocess)
syncr.syncronizer()
