# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


#question1
img = cv2.imread("dog.jpg", cv2.IMREAD_UNCHANGED)
img_info = img.shape
rows = img_info[0]
cols = img_info[1]
dst=np.zeros((rows,cols,1),np.uint8)
for i in range(rows):
    for j in range(cols):
        (b,g,r)=img[i][j]
        dst[i,j] = 0.299*int(r)+0.587*int(g)+0.114*int(b)
cv2.imwrite("dog_image I.jpg",dst)
cv2.waitKey()
cv2.destroyAllWindows()




