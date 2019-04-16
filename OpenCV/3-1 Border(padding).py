import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import publicmethod

img = cv.imread('Pictures/ts1.jpg')

reflect = cv.copyMakeBorder(img, 0, 100, 100, 100, cv.BORDER_REFLECT)
print(reflect.shape)
reflect = publicmethod.channel_change([reflect])
plt.imshow(reflect[0], 'gray')
plt.show()