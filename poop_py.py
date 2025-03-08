import numpy as np
import cupy as cp
import time
import bettercam

#hwc 1440,2560,3
camera = bettercam.create(nvidia_gpu= True)

# np_arr = camera.grab()
# print(np_arr.shape)
cp_arr = camera.grab()
# print(cp_arr.shape)
# del camera
# cp_arr = cp.asarray(camera.grab())







