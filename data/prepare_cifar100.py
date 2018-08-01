"""
This file is for preparing cifar10 and extracting it from the binary files

# Please first download cifar100 dataset and extract it in data folder here!!
# Then run this script to prepare the data of cifar100

- Generates numpys
- Generates images
- Generates tfrecords
"""
import os

import numpy as np
import imageio
import pickle
from tqdm import tqdm

import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def save_imgs_to_disk(path, arr, file_names):
    for i, img in tqdm(enumerate(arr)):
        imageio.imwrite(path + file_names[i], img, 'PNG-PIL')


def save_numpy_to_disk(path, arr):
    np.save(path, arr)


def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())


def main():
    dic_train, dic_test = unpickle('cifar-100-python/train'), unpickle('cifar-100-python/test')
    for key, val in dic_train.items():
        print(key)

    x_train = dic_train[b'data']
    x_test = dic_test[b'data']

    x_train_filenames = dic_train[b'filenames']
    x_test_filenames = dic_test[b'filenames']

    y_train = np.array(dic_train[b'fine_labels'], np.int32)
    y_test = np.array(dic_test[b'fine_labels'], np.int32)

    # Reshape and transposing the numpy array of the images
    x_train = np.transpose(x_train.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))
    x_test = np.transpose(x_test.reshape((-1, 3, 32, 32)), (0, 2, 3, 1))

    x_train_len = x_train.shape[0]
    x_test_len = x_test.shape[0]

    print(x_train.shape)
    print(x_train.dtype)
    print(y_train.shape)
    print(y_train.dtype)
    print(x_test.shape)
    print(x_test.dtype)
    print(y_test.shape)
    print(y_test.dtype)

    # exit(0)

    if not os.path.exists('cifar-100-python/imgs/'):
        os.makedirs('cifar-100-python/imgs/')

    for i in range(x_train_len):
        x_train_filenames[i] = 'imgs/' + str(x_train_filenames[i].decode('ascii'))
    for i in range(x_test_len):
        x_test_filenames[i] = 'imgs/' + str(x_test_filenames[i].decode('ascii'))

    # Save the filename of x_train and y_train
    with open('cifar-100-python/x_train_filenames.pkl', 'wb') as f:
        pickle.dump(x_train_filenames, f)
    with open('cifar-100-python/x_test_filenames.pkl', 'wb') as f:
        pickle.dump(x_test_filenames, f)

    print("FILENAMES OF IMGS saved successfully")

    print("Saving the imgs to the disk..")

    save_imgs_to_disk('cifar-100-python/', x_train, x_train_filenames)
    save_imgs_to_disk('cifar-100-python/', x_test, x_test_filenames)

    print("IMGS saved successfully")

    print("Saving the numpys to the disk..")

    save_numpy_to_disk('cifar-100-python/x_train.npy', x_train)
    save_numpy_to_disk('cifar-100-python/y_train.npy', y_train)
    save_numpy_to_disk('cifar-100-python/x_test.npy', x_test)
    save_numpy_to_disk('cifar-100-python/y_test.npy', y_test)

    print("Numpys saved successfully")

    print("Saving the data numpy pickle to the disk..")

    # SAVE ALL the data with one pickle
    with open('cifar-100-python/data_numpy.pkl', 'wb')as f:
        pickle.dump({'x_train': x_train,
                     'y_train': y_train,
                     'x_test': x_test,
                     'y_test': y_test,
                     }, f)

    print("DATA NUMPY PICKLE saved successfully..")

    print('saving tfrecord..')

    save_tfrecord_to_disk('cifar-100-python/train.tfrecord', x_train, y_train)
    save_tfrecord_to_disk('cifar-100-python/test.tfrecord', x_test, y_test)

    print('tfrecord saved successfully..')


if __name__ == '__main__':
    main()
