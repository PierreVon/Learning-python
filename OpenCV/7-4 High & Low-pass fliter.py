import cv2 as cv
import numpy as np
from publicmethod import plot, channel_change

img = cv.imread('Pictures/ts2.jpg', 0)

# Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

rows, cols = img.shape
center_row, center_col = rows//2, cols//2

high = fshift.copy()
high[center_col-30:center_col+30, center_row-30: center_row+30] = 0
f_shift = np.fft.ifftshift(high)
high_pass = np.fft.ifft2(f_shift)
high_pass = np.abs(high_pass)

low = np.zeros(fshift.shape, np.uint8)
low[center_col-100:center_col+100, center_row-100:center_row+100] = 1
low = fshift * low
f_shift = np.fft.ifftshift(low)
low_pass = np.fft.ifft2(f_shift)
low_pass = np.abs(low_pass)

images = [img, high_pass, low_pass]
titles = ['Original Image', 'High Pass Fliter', 'Low Pass Fliter']

plot(images,titles, cols=1)