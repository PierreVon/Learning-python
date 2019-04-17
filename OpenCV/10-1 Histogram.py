import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from publicmethod import plot, channel_change

img = cv.imread('Pictures/ts2.jpg',0)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
