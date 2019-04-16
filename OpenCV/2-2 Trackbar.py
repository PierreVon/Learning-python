import  cv2 as cv
import numpy as np


def nothing(x):
    pass


img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('Color')

cv.createTrackbar('R','Color',0,255,nothing)
cv.createTrackbar('G','Color',0,255,nothing)
cv.createTrackbar('B','Color',0,255,nothing)

switch = '0 : OFF \n1 : ON'

cv.createTrackbar(switch,'Color', 0, 1, nothing)

while(1):
    cv.imshow('Color', img)
    k = cv.waitKey(1)
    if k == 27:
        break

    r = cv.getTrackbarPos('R', 'Color')
    g = cv.getTrackbarPos('G', 'Color')
    b = cv.getTrackbarPos('B', 'Color')
    s = cv.getTrackbarPos(switch, 'Color')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv.destroyAllWindows()