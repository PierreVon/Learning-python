import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Pictures/ts2.jpg',0)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
