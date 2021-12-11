import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test1.jpg',0)
plt.subplot(2,2,1),plt.imshow(img,'gray')
#display histogram
plt.subplot(2,2,2),plt.hist(img.ravel(),256,[0,256])


#histogram equalization
equ = cv2.equalizeHist(img)
plt.subplot(2,2,3),plt.imshow(equ,'gray')
#display histogram
plt.subplot(2,2,4),plt.hist(equ.ravel(),256,[0,256])
plt.show()

