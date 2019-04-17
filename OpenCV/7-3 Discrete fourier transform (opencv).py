import cv2 as cv
import numpy as np
from publicmethod import plot, channel_change

img = cv.imread('Pictures/ts2.jpg', 0)

# dtf has two channels, one for real, the other for complex
dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

images = [img, magnitude_spectrum]
titles = ['Original Image', 'DFT Magnitude spectrum']

plot(images,titles, cols=1)