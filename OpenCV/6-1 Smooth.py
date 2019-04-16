import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from publicmethod import plot, channel_change

# sum of kernel of smooth must equal to 1
img = cv.imread('Pictures/ts1.jpg')

# 2D convolution
kernel = np.ones((3,3), np.float32)/9
two_convolution = cv.filter2D(img, -1, kernel)

# Mean
mean = cv.blur(img, (3,3))

# Gaussian
guassion = cv.GaussianBlur(img, (3,3), 0.2)

# Median
median = cv.medianBlur(img, 5)

# Bilateral
bilateral = cv.bilateralFilter(img, 5, 75, 75)

images = [img, two_convolution, mean, guassion, median, bilateral]
titles = ['Original Image', '2D convolution', 'Mean', 'Gaussian', 'Median', 'Bilateral']

images = channel_change(images)
plot(images,titles)