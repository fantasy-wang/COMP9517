import numpy as np 
import cv2

# read an image
img = cv2.imread('test1.jpg',1)

# display the image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# print image dimension
print(img.shape)

# print a pixel value
px = img[100,100]
print(px)

# get an ROI region
roi = img[100:120, 100:120]

# replace another region in the image with the selected ROI
img[180:200, 180:200] = roi

# display the modified image
cv2.imshow('image2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('res_1.png',img)

