import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import channel

img = cv.imread('Pictures/ts1.jpg')

reflect = cv.copyMakeBorder(img, 0, 100, 100, 100, cv.BORDER_REFLECT)
reflect = channel.channel_change(reflect)
plt.imshow(reflect, 'gray')
plt.show()