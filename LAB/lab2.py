# Template for lab02 task 3

import cv2
import math
import numpy as np
import sys
from scipy import ndimage
import matplotlib.pyplot as plt

class SiftDetector():
    def __init__(self, norm="L2", params=None):
        self.detector=self.get_detector(params)
        self.norm=norm

    def get_detector(self, params):
        if params is None:
            params={}
            params["n_features"]=0
            params["n_octave_layers"]=3
            params["contrast_threshold"]=0.31
            params["edge_threshold"]=10
            params["sigma"]=1.6

        detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.31)

        return detector

# Rotate an image
#
# image: image to rotate
# x:     x-coordinate of point we wish to rotate around
# y:     y-coordinate of point we wish to rotate around
# angle: degrees to rotate image by
#
# Returns a rotated copy of the original image
def rotatee(image, x, y, angle):
    h,w = image.shape[:2]
    M_1 = cv2.getRotationMatrix2D((x,y), angle,1)
    rotated = cv2.warpAffine(img, M_1, (w, h))
    return rotated


# Get coordinates of center point.
#
# image:  Image that will be rotated
# return: (x, y) coordinates of point at center of image
def get_img_center(image):
    h,w = image.shape[:2]
    center = (w//2, h//2)
    return center

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    # Read image with OpenCV and convert to grayscale
    img = cv2.imread('C:\\Users\\56915\\Desktop\\road_sign.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Initialize SIFT detector
    sift = SiftDetector().detector
    kp = sift.detect(gray, None)


    # Store SIFT keypoints of original image in a Numpy array
    kp, des = sift.compute(gray, kp)
    kp_np = np.array(kp)
    
    # Rotate around point at center of image.
    x,y = get_img_center(img)

    # Degrees with which to rotate image
   
    
    # Number of times we wish to rotate the image

    
    #img_rotate = rotatee(img, x,y, -90)
    img_rotate = ndimage.rotate(img,-90)
    
    gray_rotate = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)
    
    
    # BFMatcher with default params
    kp1, des1 = sift.detectAndCompute(gray, None)
    kp2, des2 = sift.detectAndCompute(gray_rotate, None)
    
    #print (np.array(kp1).shape)
    #print (np.array(kp2).shape)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    
    img3 = cv2.drawMatchesKnn(gray,kp1,gray_rotate,kp2,good,None,flags=2)
    
    
    #cv_show('img3',img3)
    cv2.imshow("img3",img3)
    cv2.imwrite('C:\\Users\\56915\\Desktop\\road_sign90.jpg',img3)
     
    #cv2.imwrite('result_3.jpg',img3)

        # Rotate image
        
        # Compute SIFT features for rotated image
        # Apply ratio test
        


        # cv2.drawMatchesKnn expects list of lists as matches.