# -*- coding: utf-8 -*-


import cv2
import numpy as np
import  matplotlib.pyplot as plt

j_img = cv2.imread("7light_rail_image J.jpg", cv2.IMREAD_UNCHANGED)
c_img = cv2.imread("light_rail.jpg", cv2.IMREAD_UNCHANGED)

b,g,r = cv2.split(c_img)
listb = []
listg = []
listr = []

size = 7
height = j_img.shape[0]
width = j_img.shape[1]
for i in range(size,height-size):
    for j in range(size,width-size):
        temp_j = j_img[i-size:i+size,j-size:j+size]
        temp = j_img[i,j]
        for m in range(i-size,i+size):
            for n in range(j-size,j+size):
                if i!=m and j!=n:
                    temp1 = j_img[m,n]
                    if temp == temp1:
                        listb.append(int(b[m,n]))
                        listg.append(int(g[m,n]))
                        listr.append(int(r[m,n]))
        mean_b = int(np.mean(listb))
        mean_g = int(np.mean(listg))
        mean_r = int(np.mean(listr))
        c_img[i,j] = (mean_b,mean_g,mean_r)
cv2.imshow("final",c_img)
cv2.imwrite("7_3_light_rail.jpg",c_img)
cv2.waitKey()
cv2.destroyAllWindows()                    

                    

