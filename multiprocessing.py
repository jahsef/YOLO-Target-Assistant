from multiprocessing import Process, Queue
from ultralytics import YOLO
import dxcam
import time
import logging
import threading
import keyboard
import win32api
import win32con
import math
import cv2
logging.getLogger('ultralytics').setLevel(logging.ERROR)

#main thread: 
#   init other threads/multiprocessing (inference, screen cap, input detection)
#   processes inference information
#   if toggled moves mouse
#   processes frame time / frame rate data

#screen cap thread:
#   at start of thread ini camera
#   capture frame
#   pass to inference thread somehow (global variable/ something thread safe???)

#inference thread:
#   wait for screen cap to pass frame (probably need a timer to check if screen cap taking too long)
#   start inference
#   pass inference results back to main thread
#   tell screen cap thread to capture again
#   optional: could screen cap on a delay (in middle of inference somehow) to reduce latency on mouse movements
#   pass results back to main thread for parsing/target selection

#input detection thread:
#   watch for user input to invert active boolean
