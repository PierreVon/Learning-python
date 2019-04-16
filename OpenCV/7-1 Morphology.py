import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from publicmethod import plot, channel_change

# sum of kernel of smooth must equal to 1
img = cv.imread('Pictures/pikachu.png')
kernel = np.ones((5,5), np.uint8)

# Erode
# when numbers in the area of origin picture covered by kernel are all 1
# the center number of the area stay still
# or it changes to 0
erode = cv.erode(img, kernel, iterations=1)

# Dilation
dilation = cv.dilate(img, kernel, iterations=1)

# Opening, erode then dilate
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

# Closing, dilate then erode
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

# Gradient, dilate - erode
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

images = [img, erode, dilation, opening, closing, gradient]
titles = ['Original Image', 'Erode', 'Dilation', 'Opening', 'Closing', 'Gradient']

images = channel_change(images)
plot(images,titles)