import cv2 as cv
import matplotlib.pyplot as plt
from math import ceil


def channel_change(img):
    for i in range(len(img)):
        b, g, r = cv.split(img[i])
        img[i] = cv.merge([r, g, b])
    return img


def plot(images, titles, cols = 2):
    total = len(images)
    rows = ceil(total/cols)
    for i in range(total):
        plt.subplot(cols, rows, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
    plt.show()
