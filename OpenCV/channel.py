import cv2 as cv


def channel_change(img):
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    return img

