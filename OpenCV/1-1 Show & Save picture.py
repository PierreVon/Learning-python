import numpy as np
import cv2 as cv

img = cv.imread('Pictures/ts1.jpg', 0)
# makes window scalable
cv.namedWindow('Taylor Swift', cv.WINDOW_NORMAL)

cv.imshow('Taylor Swift', img)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
elif k == ord('s'):
    cv.imwrite('ts1.png', img)
    cv.destroyAllWindows()
