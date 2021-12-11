import tensorflow as tf
import numpy as np
import cv2
import load
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
train_images = []
train_lables = []
grid_train_images = []
grid_label_images = []
train_data, test_data = load.load_data()
grid_train_data = {}
grid_train_label = {}

# health soft head hae mic
# 0 1 2 3 4


def gene_grid(origin_images, masks, label, h=0):
    num = 6000
    len_images = len(origin_images)
    grid_images = []
    grid_labels = []
    pos = []
    for i in range(len_images):
        for j in range(len(masks[i])):
            for k in range(len(masks[i][j])):
                if masks[i][j][k] != 0 and (k >= 16 and k <= 240 and j >= 16 and j <= 240):
                    pos.append((i, j, k))
    len_pos = len(pos)
    if num > len_pos:
        num = len_pos
    if h == 1:
        num = 12000
    index = random.sample(range(len_pos), num)
    num = 6000
    for i in index:
        n3 = pos[i][0]
        n1 = pos[i][1]
        n2 = pos[i][2]
        new = origin_images[n3][n1 - 16:n1 + 16, n2 - 16:n2 + 16, :]
        grid_images.append(new)
        grid_labels.append(label)
    return grid_images, grid_labels


def gene_test(origin_image):
    grid_images = []
    for i in range(16, 240):
        for j in range(16, 240):
            new = origin_image[i - 16:i + 16, j - 16:j + 16, :]
            grid_images.append(new[np.newaxis, :])
    return grid_images


test = gene_test(test_data['origin'][0])
test_grid_ima = np.squeeze(np.vstack(test))
plt.switch_backend('agg')
grid_train_data['soft'], grid_train_label['soft'] = gene_grid(
    train_data['origin'], train_data['soft'], 1)
grid_train_data['hard'], grid_train_label['hard'] = gene_grid(
    train_data['origin'], train_data['hard'], 2)
grid_train_data['hae'], grid_train_label['hae'] = gene_grid(
    train_data['origin'], train_data['hae'], 3)
grid_train_data['mic'], grid_train_label['mic'] = gene_grid(
    train_data['origin'], train_data['mic'], 4)
grid_train_data['heal'], grid_train_label['heal'] = gene_grid(
    train_data['origin'], train_data['heal'], 0, 1)
unum = ['soft', 'hard', 'hae', 'mic', 'heal']
temp = []
for i in unum:
    temp += grid_train_data[i]
for i in range(len(temp)):
    temp[i] = temp[i][np.newaxis, ...]
train_datasets = np.squeeze(np.vstack(temp))
temp = []
for i in unum:
    temp.append(grid_train_label[i])
train_label = np.squeeze(np.hstack(temp))
temp = random.sample(range(len(train_datasets)), len(train_datasets))
train_datasets = train_datasets[temp]
train_label = train_label[temp]
train_datasets_val = train_datasets[-3000:]
train_label_val = train_label[-3000:]
train_data_new = train_datasets[:-3000]
train_label_new = train_label[:-3000]
train_datasets_val = train_datasets_val / 255
train_data_new = train_data_new / 255
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        32, (4, 4), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same',
                           kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same',
                           kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same',
                           kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same',
                           kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1000, activation='relu',
                          kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='relu',
                          kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_new, train_label_new, batch_size=6000, epochs=600,
          validation_data=(train_datasets_val, train_label_val))
test = gene_test(test_data['origin'][9])
test_grid_ima = np.squeeze(np.vstack(test))
test_grid_ima = test_grid_ima / 255
ans = model.predict(test_grid_ima)
ans = np.argmax(ans, axis=1)
ans = np.reshape(ans, (240 - 16, 240 - 16))
ans = np.squeeze(ans)
ans = np.array(ans, dtype='uint8')
ans = cv2.resize(ans, (4000, 2828))
plt.imshow(ans)
plt.show()
plt.savefig('ans.png')
model.save('aaa.h5')