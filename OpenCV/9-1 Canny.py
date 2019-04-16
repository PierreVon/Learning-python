import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Pictures/ts2.jpg', 0)

# Canny
# 1. smooth
# 2. calculate gradient of x, y, edge gradient = sqrt(x^2 + y^2)
# 3. scan picture, fine maximum value
# 4. drop value under minVal, if value threshold between minVal and maxVal and connect to same value
#    higher than maxVal, keep it
canny = cv.Canny(img, 100, 200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()