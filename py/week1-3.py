import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('noisyImg_1.jpg',1)
#convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),3)

#Gaussian filtering v2
G = cv2.getGaussianKernel(5,3)
GM = G*G.T;
print(G)
print(GM)
blur2 = cv2.filter2D(img,-1,GM)

#show original and filtered images
plt.subplot(1,3,1),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(blur2),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()

