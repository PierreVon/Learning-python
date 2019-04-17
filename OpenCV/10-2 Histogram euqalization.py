import cv2 as cv
import numpy as np
from publicmethod import plot, channel_change

# only in  gray scale
img = cv.imread('Pictures/ts2.jpg', 0)

# naive equalization
equalization = cv.equalizeHist(img)

# clahe
# divide picture with tiles normally 8*8
# equalize each tiles
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_equalization = clahe.apply(img)

images = [img, equalization, clahe_equalization]
titles = ['Original Image', 'Equalization', 'CLAHE']

plot(images,titles)