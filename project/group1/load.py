import numpy as np
import os
import numpy
from PIL import Image
from matplotlib import pyplot as plt
import cv2

msize = (256, 256)


def load_datasets(low, high, path, suffix):
    ims = []
    for i in range(low, high):
        temp = str(i)
        if len(temp) == 1:
            temp = '0' + temp
        try:
            im = Image.open(path + temp + suffix)
            im = np.array(im)
            im = cv2.resize(im, (256, 256))
        except IOError:
            im = np.zeros(msize)
        ims.append(im)
    return ims


a = load_datasets(
    1, 55, 'Data_Group_Component_Task_1/Train/masks_Microaneurysms/IDRiD_', '_MA.tif')
a = np.array(a[0])
a.shape

train_data = {}
test_data = {}


def load_data():
    train_data['origin'] = load_datasets(
        1, 55, 'Data_Group_Component_Task_1/Train/original_retinal_images/IDRiD_', '.jpg')
    train_data['soft'] = load_datasets(
        1, 55, 'Data_Group_Component_Task_1/Train/masks_Soft_Exudates/IDRiD_', '_SE.tif')
    train_data['hard'] = load_datasets(
        1, 55, 'Data_Group_Component_Task_1/Train/masks_Hard_Exudates/IDRiD_', '_EX.tif')
    train_data['hae'] = load_datasets(
        1, 55, 'Data_Group_Component_Task_1/Train/masks_Haemorrhages/IDRiD_', '_HE.tif')
    train_data['mic'] = load_datasets(
        1, 55, 'Data_Group_Component_Task_1/Train/masks_Microaneurysms/IDRiD_', '_MA.tif')
    temp = []
    for i in range(len(train_data['origin'])):
        temp.append(train_data['soft'][i] + train_data['hard']
                    [i] + train_data['hae'][i] + train_data['mic'][i])
        a = np.zeros(msize)
        a[temp[-1] == 0] = 1
        temp[-1] = a
    train_data['heal'] = temp

    test_data['origin'] = load_datasets(
        55, 82, 'Data_Group_Component_Task_1/Test/original_retinal_images/IDRiD_', '.jpg')
    test_data['soft'] = load_datasets(
        55, 82, 'Data_Group_Component_Task_1/Test/masks_Soft_Exudates/IDRiD_', '_SE.tif')
    test_data['hard'] = load_datasets(
        55, 82, 'Data_Group_Component_Task_1/Test/masks_Hard_Exudates/IDRiD_', '_EX.tif')
    test_data['hae'] = load_datasets(
        55, 82, 'Data_Group_Component_Task_1/Test/masks_Haemorrhages/IDRiD_', '_HE.tif')
    test_data['mic'] = load_datasets(
        55, 82, 'Data_Group_Component_Task_1/Test/masks_Microaneurysms/IDRiD_', '_MA.tif')
    temp = []
    for i in range(len(test_data['origin'])):
        temp.append(test_data['soft'][i] + test_data['hard']
                    [i] + test_data['hae'][i] + test_data['mic'][i])
        a = np.zeros(msize)
        a[temp[-1] == 0] = 1
        temp[-1] = a
    test_data['heal'] = temp

    return train_data, test_data
