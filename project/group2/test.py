from matplotlib import pyplot as plt
import scipy.ndimage as nd
import load_data
import numpy as np
import cv2
import tensorflow as tf
import random
train_data, test_data = load_data.load_task2_dataset()
# preprocessing
train_data['segmented'] = list(map(lambda x: x/255, train_data['segmented']))
test_data['segmented'] = list(map(lambda x: x/255, test_data['segmented']))
train = []
test = []
for i in range(20):
    train.append(np.load('train{}.npy'.format(i)))
    test.append(np.load('test{}.npy'.format(i)))
a = np.array([[1, 2, 3, 4]])
b = np.array([[1, 0, 1, 0]])
train_data_flat = [np.reshape(i, (-1, 7)) for i in train]
test_data_flat = [np.reshape(i, (-1, 7)) for i in test]
train_data_flat = np.squeeze(np.vstack(train_data_flat))
train_label_flat = [np.reshape(i, (-1, 1)) for i in train_data['segmented']]
train_label_flat = np.squeeze(np.vstack(train_label_flat))
train_label_p = train_label_flat[train_label_flat == 1]
train_label_n = train_label_flat[train_label_flat == 0]
train_data_p = train_data_flat[train_label_flat == 1]
train_data_n = train_data_flat[train_label_flat == 0]

len_p = len(train_data_p)
len_n = len(train_data_n)
sam = (random.sample(range(len_n), len_p))
train_data_n_select = train_data_n[sam]
train_label_n_select = train_label_n[sam]
train_data_new = np.squeeze(np.vstack([train_data_n_select, train_data_p]))
train_label_new = np.squeeze(np.hstack([train_label_n_select, train_label_p]))
sam = (random.sample(range(len(train_data_new)), len(train_data_new)))


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_data_val = train_data_new[-100000:]
train_label_val = train_label_new[-100000:]
train_data_new = train_data_new[:-100000]
train_label_new = train_label_new[:-100000]
print(10)
model.fit(train_data_new, train_label_new, batch_size=20000,
          epochs=100, validation_data=(train_data_val, train_label_val))
aa = model.predict(np.reshape(test[0], (-1, 7)))
aa = np.argmax(aa, axis=1)
aa = np.reshape(aa, [test[0].shape[0], test[0].shape[1]])
plt.imshow(aa)
plt.show()
model.save('model.h5')
