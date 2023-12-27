import argparse
from PyQt6.QtWidgets import QApplication
import sys
from Qt import App

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--path", help="path and name of output video")
argParser.add_argument("-s", "--no_stack", help="If you dont want to record and check exposure", action = "store_true")

args = argParser.parse_args()

app = QApplication(sys.argv)
ex = App(args)
sys.exit(app.exec())