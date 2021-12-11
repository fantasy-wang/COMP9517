# -*- coding: utf-8 -*-
import cv2
import numpy as np
import  matplotlib.pyplot as plt

img = cv2.imread("dog_image I.jpg", cv2.IMREAD_UNCHANGED)
new_img = img

#define a size
size = 21
height = img.shape[0]
width = img.shape[1]

#make a border of image
new = cv2.copyMakeBorder(img,size,size,size,size,cv2.BORDER_REFLECT)

#loop every point
for i in range(size,height):
    for j in range(size,width):
        #define a windows
        temp = new[i-size:i+size,j-size:j+size]
        hist = cv2.calcHist([temp],[0],None,[256],[0,256])
        result = hist.ravel()
        result_max = np.max(result)
        result_max_num = np.where(result==result_max)
        new_img[i-size,j-size] = result_max_num[0][0]
cv2.imwrite("21dog_image J.jpg",new_img)
cv2.waitKey()
cv2.destroyAllWindows()





