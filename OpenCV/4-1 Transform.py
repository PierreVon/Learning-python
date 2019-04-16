import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from publicmethod import plot

img = cv.imread('Pictures/ts2.jpg', 0)
height,width=img.shape[:2]

# Resize
shrink = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)

enlargement = cv.resize(img,(height//2,width//2), interpolation=cv.INTER_AREA)

# Translation
# [[1, 0, tx], [0, 1, ty]]
move = np.float32([[1, 0, 50], [0, 1, 100]])
translation = cv.warpAffine(img,move,(width, height))

# Rotation
move = cv.getRotationMatrix2D((width/2, height/2), 45, 0.6)
rotation = cv.warpAffine(img, move, (width, height))

# Affine
# linea paralleled in origin picture parallel in transformed picture
# it requires three points
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
move = cv.getAffineTransform(pts1,pts2)
affine = cv.warpAffine(img, move, (width, height))

# Perspective
pts1 = np.float32([[250,250],[125,500],[500,250],[375,500]])
pts2 = np.float32([[0,0],[0,300],[300,0],[300,300]])
move = cv.getPerspectiveTransform(pts1, pts2)
perspective = cv.warpPerspective(img, move, (width, height))

images = [img, shrink, enlargement, translation, rotation, affine, perspective]
titles = ['Original Image', 'Shrink', 'Enlargement', 'Translation', 'Rotation', 'Affine', 'Perspective']

plot(images, titles, cols=3)
