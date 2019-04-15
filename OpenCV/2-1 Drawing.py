import numpy as np
import cv2 as cv

# 3 stands for 3 channels
img = np.zeros((512, 512, 3), np.uint8)

# line, start end
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# rectangle, left-top, right-bottom
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# circle, center radius
cv.circle(img, (255,255), 40, (0, 0, 255), -1)

# polygon,
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(img, [pts], True, (0, 255, 255),5)

# text
font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'PierreVon', (10, 500), font, 3, (255, 255, 255), 2, cv.LINE_AA)

cv.imshow('Canvas',img)
cv.waitKey(0)
