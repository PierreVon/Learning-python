import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('Pictures/ts1.jpg')
cv.imshow('Taylor Swift', img)
b, g, r = cv.split(img)
img = cv.merge([r, g, b])
plt.imshow(img)
plt.show()
k = cv.waitKey(0)
