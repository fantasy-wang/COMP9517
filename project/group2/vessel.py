from matplotlib import pyplot as plt
import scipy.ndimage as nd
import load_data
import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(15, activation='relu', input_shape=(7,),
                          tf.keras.layers.Dense(15, activation='relu'),
                          tf.keras.layers.Dense(15, activation='relu'),
                          tf.keras.layers.Dense(15, activation='softmax'))
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


try:
    # '.' if the path is to current folder
    os.chdir(os.path.join(os.getcwd(), '/home/kf/桌面/group'))
    print(os.getcwd())
except:
    pass

train_data, test_data = load_data.load_task2_dataset()
# preprocessing
train_data['masks'] = list(map(lambda x: x/255, train_data['masks']))
test_data['masks'] = list(map(lambda x: x/255, test_data['masks']))


def mask_set(tp=0):
    if tp == 0:
        train_data['original'] = list(map(
            lambda x, y: x * (y), train_data['original'], train_data['masks']))
        test_data['original'] = list(
            map(lambda x, y: x*(y), test_data['original'], test_data['masks']))
    if tp == 1:
        def ave_set(ip, mask):
            _mask = 1 - mask
            ave = np.average(ip, weights=mask)
            ip += _mask * ave
            return ip
        train_data['original'] = list(
            map(ave_set, train_data['original'], train_data['masks']))
        test_data['original'] = list(
            map(ave_set, test_data['original'], test_data['masks']))


def blur_data(func, ksize, tp=0):
    if tp == 0:
        train_data['original'] = list(
            map(lambda x: func(x, ksize), train_data['original']))
        test_data['original'] = list(
            map(lambda x: func(x, ksize), test_data['original']))
    if tp == 1:
        train_data['original'] = list(
            map(lambda x: x-func(x, ksize)+128, train_data['original']))
        test_data['original'] = list(
            map(lambda x: x-func(x, ksize)+128, test_data['original']))


lenx, leny = train_data['original'][0].shape
vec_w1 = 4
vec_w2 = 8


def window_mat(x, y, ksize, mat):
    lmat = np.zeros((2*ksize+1, 2*ksize+1))
    for i in range(x-ksize, x+ksize+1):
        for j in range(y - ksize, y + ksize + 1):
            if i < 0 or j < 0 or i >= lenx or j >= leny:
                lmat[i + ksize - x][j + ksize - y] = 128
            else:
                lmat[i + ksize - x][j + ksize - y] = mat[i][j]
    return lmat


# test_mat = np.zeros((20, 20))
# count = 0
# for i in range(20):
#     for j in range(20):
#         test_mat[i][j] = count
#         count += 1
# print(window_mat(19, 1, 5, test_mat))

GaussianKernal = cv2.getGaussianKernel(17, 1.7)


def cal_vec(ori_mat, ve_mat):
    mat = np.zeros((lenx, leny, 7))
    for i in range(lenx):
        for j in range(leny):
            lmat = window_mat(i, j, vec_w1, ori_mat)
            minI = np.min(lmat)
            maxI = np.max(lmat)
            meanI = np.mean(lmat)
            stdI = np.std(lmat)
            mat[i][j][0] = ori_mat[i][j] - minI
            mat[i][j][1] = maxI - ori_mat[i][j]
            mat[i][j][2] = ori_mat[i][j] - meanI
            mat[i][j][3] = stdI
            mat[i][j][4] = ori_mat[i][j]
            lmat = window_mat(i, j, vec_w2, ve_mat)
            lmat = lmat * GaussianKernal
            ict00 = np.ones(lmat.shape)
            ict11 = np.ones(lmat.shape)
            ict10 = np.ones(lmat.shape)
            ict01 = np.ones(lmat.shape)
            ict02 = np.ones(lmat.shape)
            ict20 = np.ones(lmat.shape)
            for l in range(lmat.shape[0]):
                for m in range(lmat.shape[1]):
                    ict10[l][m] = (i - vec_w2 + l)
                    ict01[l][m] = (j - vec_w2 + m)
            m10 = np.sum(ict10*lmat)
            m01 = np.sum(ict01 * lmat)
            m00 = np.sum(lmat)
            ai = m10 / m00
            aj = m01 / m00
            mu00 = m00
            for l in range(lmat.shape[0]):
                for m in range(lmat.shape[1]):
                    ict11[l][m] = (i - vec_w2 + l - ai) * (j - vec_w2 + m - aj)
                    ict20[l][m] = (i - vec_w2 + l - ai) ** 2
                    ict02[l][m] = (j - vec_w2 + m - aj) ** 2
            eta02 = np.sum(ict02 * lmat)
            eta20 = np.sum(ict20 * lmat)
            eta11 = np.sum(ict11 * lmat)
            eta02 /= mu00 ** 2
            eta20 /= mu00 ** 2
            eta11 /= mu00 ** 2
            phi1 = eta02 + eta20
            phi2 = (eta02 + eta20) ** 2 + 4 * eta11 ** 2
            mat[i][j][5] = np.log(phi1)
            mat[i][j][6] = np.log(phi2)
    return mat


mask_set()

train_data['original'] = list(
    map(lambda x: nd.grey_opening(x, (3, 3)), train_data['original']))
test_data['original'] = list(
    map(lambda x: nd.grey_opening(x, (3, 3)), test_data['original']))

train_origin = np.array(train_data['original'])
test_origin = np.array(test_data['original'])
blur_data(cv2.blur, (3, 3))
blur_data(lambda x, ksize: cv2.GaussianBlur(
    x, ksize=ksize, sigmaX=1.8), (9, 9))

mask_set(1)
blur_data(cv2.blur, (69, 69))
train_origin = list(
    map(lambda x, y: x-y+128, train_origin, train_data['original']))
test_origin = list(
    map(lambda x, y: x - y + 128, test_origin, test_data['original']))
train_ve = list(
    map(lambda x: nd.grey_opening(-x, (16, 16))+x + 128, train_origin))
test_ve = list(map(lambda x: nd.grey_opening(-x, (16, 16))+x+128, test_origin))
# print(cal_vec(train_ve[0]))
train_feature = list(map(cal_vec, train_origin, train_ve))
test_feature = list(map(cal_vec, test_origin, test_ve))
plt.figure()
plt.imshow(train_origin[13], cmap=plt.cm.gray)
# plt.show()
plt.figure()
plt.imshow(train_ve[13], cmap=plt.cm.gray)
plt.figure()
plt.imshow((train_data['segmented'][13]), cmap=plt.cm.gray)
plt.show()


for i in range(len(train_feature)):
    np.save('train{}'.format(i), train_feature[i])
    np.save('test{}'.format(i), test_feature[i])

a = np.load('train1.npy')