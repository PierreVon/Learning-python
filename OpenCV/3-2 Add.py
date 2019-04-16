import numpy as np
import cv2 as cv

# g (x) = (1 − α)f 0 (x) + αf 1 (x)

img1 = cv.imread('Pictures/pikachu.png')
img2 = cv.imread('Pictures/ts2.jpg')
rows, cols, channels = img1.shape
img2 = cv.resize(img2, (cols, rows))

img = cv.addWeighted(img1, 0.7, img2, 0.3, 0)

cv.imshow('Taylor Swift', img)
cv.waitKey(0)
cv.destroyAllWindows()