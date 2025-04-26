import numpy as np


xyxy_arr = np.array([[100, 200, 150, 250],
                     [120, 220, 170, 270]])

x1, y1, x2, y2 = xyxy_arr[:, 0:4].T
print(x1,y1,x2,y2)
# x1 will be [100, 120] and so on, which is not what you want.