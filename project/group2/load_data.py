import numpy as np
import os
import numpy
from PIL import Image
from matplotlib import pyplot as plt

try:
    # '.' if the path is to current folder
    os.chdir(os.path.join(os.getcwd(), '/home/kf/桌面/group'))
    print(os.getcwd())
except:
    pass


def load_datasets(low, high, path, suffix):
    ims = []
    for i in range(low, high):
        temp = str(i)
        if len(temp) == 1:
            temp = '0' + temp
        im = Image.open(path+temp+suffix)
        ims.append(im)
    return ims


def green_layer(ims):
    return list(map(lambda x: (x.split()[1]), ims))


def matrix_layer(ims):
    return list(map(lambda x: np.array(x), ims))


def load_task2_dataset():
    ims_train = {'original': [], 'masks': [], 'segmented': []}
    ims_test = {'original': [], 'masks': []}
    ims_train['original'] = matrix_layer(green_layer(load_datasets(
        21, 41, 'Data_Group_Component_Task_2/Training/original_retinal_images/', '_training.tif')))
    ims_train['masks'] = matrix_layer(load_datasets(
        21, 41, 'Data_Group_Component_Task_2/Training/background_masks/', '_training_mask.gif'))
    ims_train['segmented'] = matrix_layer(load_datasets(
        21, 41, 'Data_Group_Component_Task_2/Training/blood_vessel_segmentation_masks/', '_manual1.gif'))
    ims_test['original'] = matrix_layer(green_layer(load_datasets(
        1, 21, 'Data_Group_Component_Task_2/Test/original_retinal_images/', '_test.tif')))
    ims_test['masks'] = matrix_layer(load_datasets(
        1, 21, 'Data_Group_Component_Task_2/Test/background_masks/', '_test_mask.gif'))
    ims_test['segmented'] = matrix_layer(load_datasets(
        1, 21, 'Data_Group_Component_Task_2/Test/blood_vessel_segmentation_masks/', '_manual1.gif'))
    return ims_train, ims_test


# im = Image.open(
#     'Data_Group_Component_Task_2/Training/blood_vessel_segmentation_masks/'+str(21)+'_manual1.gif')
s = load_task2_dataset()
