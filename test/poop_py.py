import numpy as np
import cupy as cp
import time
import bettercam


camera = bettercam.create(nvidia_gpu= True)

cp_arr = camera.grab()






