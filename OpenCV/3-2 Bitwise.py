import numpy as np
import cv2 as cv
from publicmethod import plot, channel_change

# g (x) = (1 − α)f 0 (x) + αf 1 (x)

img1 = cv.imread('Pictures/pikachu.png')
img2 = cv.imread('Pictures/ts2.jpg')
rows, cols, channels = img1.shape
img2 = cv.resize(img2, (cols, rows))

bitwise_and = cv.bitwise_and(img1, img2)
bitwise_or = cv.bitwise_or(img1, img2)
bitwise_not = cv.bitwise_not(img1, img2)
bitwise_xor = cv.bitwise_xor(img1, img2)

images = [bitwise_and, bitwise_or, bitwise_not, bitwise_xor]
titles = ['And', 'Or', 'Not', 'Xor']

images = channel_change(images)
plot(images,titles)