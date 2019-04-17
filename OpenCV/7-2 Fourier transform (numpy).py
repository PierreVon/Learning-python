import cv2 as cv
import numpy as np
from publicmethod import plot, channel_change

img = cv.imread('Pictures/ts2.jpg', 0)

# Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

images = [img, magnitude_spectrum]
titles = ['Original Image', 'Magnitude Spectrum']

plot(images,titles, cols=1)