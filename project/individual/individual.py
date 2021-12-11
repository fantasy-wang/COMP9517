# %%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
% matplotlib
inline
# %%
os.listdir("./")
# %%
PATH_TO_IMAGE = [os.path.join('./original_retinal_images', i) for i in os.listdir("./original_retinal_images")]
PATH_TO_OD = [os.path.join('./target_images', i) for i in os.listdir("./target_images")]
# %% md
# Display some images
# %%
from IPython.display import Image as Image_display

Image_display(PATH_TO_IMAGE[0])
# %%
# Load images
imgs = [cv2.imread(PATH, 0) for PATH in PATH_TO_IMAGE]  # all the images are 2848*4288
imgs_OD = [cv2.imread(PATH, 0) for PATH in PATH_TO_OD]  # all the images are 2848*4288
# %%
imgs_rgb = [cv2.cvtColor(cv2.imread(PATH), cv2.COLOR_BGR2RGB) for PATH in PATH_TO_IMAGE]  # all the images are 2848*4288
# %%
plt.imshow(imgs_rgb[0])
# %%
# Display gray figures
# For convenience, we convert the imgs to 256*256 to save time
img_resized = [cv2.resize(img, (800, 800)) for img in imgs]
img_OD_resized = [cv2.resize(img, (800, 800)) for img in imgs_OD]

# Display
plt.subplot(1, 2, 1)
plt.title("First Original Figure")
plt.imshow(img_resized[0], cmap='gray')
plt.subplot(1, 2, 2)
plt.title("First Target Figure")
plt.imshow(img_OD_resized[0], cmap='gray')
# %% md
# HoughCircles
# %%
for i in range(5):
    img = img_resized[i]
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=65, param2=20, minRadius=100, maxRadius=200)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #         # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %% md
# Laplacian
# %%
for i in range(5):
    img = img_resized[i]
    img = cv2.medianBlur(img, 5)

    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    plt.imshow(lap, cmap='gray')
    plt.show()
# %% md
# Sobel
# %%
for i in range(5):
    img = img_resized[i]
    img = cv2.medianBlur(img, 5)

    # Sobel边缘检测
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # x方向的梯度
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)  #

    plt.imshow(sobelCombined, cmap='gray')
    plt.show()
# %% md
# Canny
# %%
for i in range(5):
    img = img_resized[i]
    img = cv2.medianBlur(img, 5)
    canny = cv2.Canny(img, 30, 150)

    plt.imshow(canny)
    plt.show()
# %% md
# Fourier Transformation
# %%
for i in range(5):
    img = img_resized[i]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
# %%
for i in range(5):
    img = img_resized[i]
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 50, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    plt.show()


# %% md
# RegionGrow
# %%
class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


img = img_resized[0]
x, y = np.where(img == 200)
seeds = [Point(i, j) for i, j in zip(x, y)]
binaryImg = regionGrow(img, seeds, 10)

plt.imshow(binaryImg, cmap='gray')
# %% md
# c-means Fuzzy
# %%
import skfuzzy as fuzz

# %%
for i in range(5):

    img = img_resized[i]
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original')

    ncenters = 5
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img, ncenters, 2, error=0.005, maxiter=1000, init=None)
    cluster_membership = np.argmax(u, axis=0)

    x_index, y_index = np.unravel_index(np.argmax(img), img.shape)
    highest_pixel = cluster_membership[x_index]

    for j in range(ncenters):
        if j != highest_pixel:
            img[cluster_membership == j] = 0

    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Get rid of low clusters')
    plt.show()
# %%
help(fuzz.cluster.cmeans)


# %% md
# Threshold
# %%
def get_accuracy(result, original):
    intersection = (
                (np.concatenate(th1).astype(np.uint16) + np.concatenate(tar_th1).astype(np.uint16)) == 255 + 255).sum()
    Union = (np.concatenate(result) == 255).sum() + (np.concatenate(original) == 255).sum() - intersection
    #     Union = (np.concatenate(original) == 255).sum()

    return intersection / Union


accuracy_list = []

for i in range(len(img_resized)):
    img = img_resized[i]
    tar_img = img_OD_resized[i]

    ret, th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    ret, tar_th1 = cv2.threshold(tar_img, 1, 255, cv2.THRESH_BINARY)

    titles = ['Original', 'BINARY', 'tar_Original', 'tar_BINARY']
    images = [img, th1, tar_img, tar_th1]
    accuracy = get_accuracy(th1, tar_th1)
    accuracy_list.append(accuracy)

    # Using matplotlib to display
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i], fontsize=8)
        plt.xticks([]), plt.yticks([])

    plt.show()
    print(f"The accuracy of THRESH_BINARY is： {accuracy}")

# %%
np.mean(accuracy_list)