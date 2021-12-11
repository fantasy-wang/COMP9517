from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


img = cv.imread('ansel_adams.jpg', 0)
g = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
w, h = img.shape


def convoloved(img, k):
    re = np.zeros(img)
    for i in range(0, w-1):
        for j in range(0, h-1):
            re[i, j] = (g * img[i: i+3, j: j+3]).sum()
            if re[i, j] < 0:
                re[i, j] = 0
            if re[i, j] > 255:
                re[i, j] = 255
    return re

img1 = convoloved(img, g)

plt.rcParams['figure.figsize'] = (300, 300)
plt.axis('off')
# plt.show()
plt.imshow(img, 'grey')
# save
plt.savefig('D:\\9517\\ansel_adams.png')

