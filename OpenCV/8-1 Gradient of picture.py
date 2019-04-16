import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from publicmethod import plot, channel_change

img = cv.imread('Pictures/pikachu.png')

# Sobel X
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)

# Sobel Y
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# Laplacian
laplacian = cv.Laplacian(img, cv.CV_64F, ksize=5)

# important
# set ddepth as cv.CV_64F because of gradient has minus value
# uint8 will set all minus value as 0

images = [img, sobelX, sobelY, laplacian]
titles = ['Original Image', 'Sobel X', 'Sobel Y', 'Laplacian']

images = channel_change(images)
plot(images,titles)