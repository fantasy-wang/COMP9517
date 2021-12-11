import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:\\9517\\ansel_adams.jpg', 0)

F_x = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


def convolve2d(image, kernel):
    h = image.shape[1]
    w = image.shape[0]

    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    output = np.zeros(img.shape)  # convolution output
    for x in range(0, h):  # Loop over every pixel of the image
        for y in range(0, w):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + 3, x:x + 3]).sum()
            output[y, x] = 0 if output[y, x] < 0 else output[y, x]
            output[y, x] = 255 if output[y, x] > 255 else output[y, x]
    return output


img1 = convolve2d(img, F_x)

plt.rcParams['figure.figsize'] = (15, 5)
plt.axis('off')

plt.imshow(img, 'gray')
plt.savefig("D:\\9517\\temp.png")
