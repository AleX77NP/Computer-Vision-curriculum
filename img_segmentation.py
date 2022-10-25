# region-based segmentation
from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

image = plt.imread('images/1.jpeg')
print(image.shape)

gray = rgb2gray(image)
# cv2.imshow('img', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(image)

gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
    if gray_r[i] > gray_r.mean():
        gray_r[i] = 1
    else:
        gray_r[i] = 0

gray = gray_r.reshape(gray.shape[0], gray.shape[1])

cv2.imshow('img', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
