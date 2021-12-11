# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from sklearn.cluster import MeanShift

from PIL import Image

size = 100, 100
#read image and array
img = Image.open("two_halves.png")
img.thumbnail(size)
img_array = np.array(img)[:,:,:3]

#extract b,g,r
b = np.array(img)[:,:,0]
g = np.array(img)[:,:,1]
r = np.array(img)[:,:,2]

#print(b)


'''#extract b,g,r
b = img_array[:,:,0]
g = img_array[:,:,1]
r = img_array[:,:,2]
'''
#print(b)
#get one shape from b,g,r
b_shape = b.shape

#reshape
re = img_array.reshape(3,-1)

#flatten each channel
'''b = b.ravel()   
g = g.ravel()
r = r.ravel()
b = b.flatten()
g = g.flatten()
r = r.flatten()
'''



#combine
img_flat = img_array.flatten()
colour_samples = img_flat.reshape(3,-1)

colour_samples = b
#= np.dstack((r,g,b))
#colour_samples = colour_samples.flatten()

#print(colour_samples)



#print(colour_samples)
#meanshift

ms_clf = MeanShift(bin_seeding=True)
ms_labels = ms_clf.fit_predict(colour_samples)

print(ms_labels)







































