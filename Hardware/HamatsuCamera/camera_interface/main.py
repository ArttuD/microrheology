import argparse
import sys
import os
from pyQt_class import *

parser = argparse.ArgumentParser(description='Microrheology test setup.')
parser.add_argument('--path', '-p', required = False, type = str, default = r'.\test', help='Save path for all the files, creates folder etc')

args = parser.parse_args()
isExist = os.path.exists(args.path)

if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(args.path)
   print("The new directory is created: ", args.path )

#launch graph
pymain(args)