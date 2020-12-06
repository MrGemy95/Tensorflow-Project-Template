import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from bunch import Bunch
import math
import cv2
import matplotlib.pyplot as plt

class DataSet:
    def __init__(self, config):
        self.config = config
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=config.data_train)
        #get the number of all training and validtion images
        self.train_img_cnt = x_train.shape[0]
        self.val_img_cnt   = x_test.shape[0]

        # select the minimum batch counter
        train_batch_num = int(math.ceil(float(self.train_img_cnt) / config.batch_size))
        if config.num_iter_per_epoch > train_batch_num:
            config.num_iter_per_epoch      = train_batch_num
            self.config.num_iter_per_epoch = train_batch_num

        with tf.name_scope("dataset"):
            #generate the dataset
            train_dataset, val_dataset = self.gen_dataset(x_train, y_train, x_test, y_test, self.train_img_cnt, config.batch_size)
            #data access iterator,both for train and val
            self.train_iter = iter(train_dataset)
            self.val_iter = iter(val_dataset)
            # should be called in each epoch
            #self.train_init_op = self.train_iter.make_initializer(train_dataset)
            #self.val_init_op   = self.train_iter.make_initializer(val_dataset)

    #use the tf.Dataset to prepare the data
    def get_next(self, data = "train"):
        with tf.name_scope("get_next"):
            if data == "train":
                image, label = next(self.train_iter)
            else:
                image, label = next(self.val_iter)
            #add some image processing with tensorflow here
            x = tf.expand_dims(image, axis = -1)
            x = tf.cast(x,dtype=tf.float32)
            x = x / 255.0

            y = tf.one_hot(label, depth=10,axis=-1)
            y = tf.cast(y,dtype=tf.int32)

        return x, y

    def init_train_iter(self, sess):
        sess.run(self.train_init_op)

    def init_val_iter(self, sess):
        sess.run(self.val_init_op)

    def gen_dataset(self, x_train, y_train, x_test, y_test, train_cnt, batch_size):
        #input a txt file which contain the image and lable information
        train_dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_train)),
             tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_train)))
        )
        train_dataset = train_dataset.shuffle(train_cnt)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.repeat()
        #train_dataset = train_dataset.prefetch(4)

        val_dataset = tf.data.Dataset.zip(
            (tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_test)),
             tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(y_test)))
        )
        val_dataset = val_dataset.batch(5)
        #val_dataset = val_dataset.prefetch(4)
        return train_dataset, val_dataset

if __name__ == "__main__":
    config_dict = {
        "data_train":"D:/job/sandbox_fvg3/multi_tp/dataset/mnist/mnist.npz",
        "num_class": 10,
        "input_h": 28,
        "input_w": 28,
        "input_c": 1,
        "batch_size": 5,
        "num_iter_per_epoch":2000000
    }
    config = Bunch(config_dict)
    data = DataSet(config)


    def show_image(img_data):
        if img_data.shape[-1] == 1:
            plt.imshow(img_data, cmap='gray')
        else:
            plt.imshow(img_data)
        plt.show()

    def show_image_cv(img_data, title = "img"):
        h = img_data.shape[0]
        w = img_data.shape[1]
        if h < 50 or w < 50:
            h = int(h * 10.0)
            w = int(w * 10.0)
        img_data = cv2.resize(img_data, (w, h))
        cv2.imshow(title, img_data)
        cv2.waitKey(2000)

    x_, y_ = data.get_next()
    print(type(x_), type(x_.numpy()))
    print(x_.shape, x_.ndim)
    print(type(y_), type(y_.numpy()))
    print(y_.shape, y_.ndim)
    for i in range(x_.shape[0]):
        show_image_cv(x_.numpy()[i,...], title="img-train")
        print(np.max(x_.numpy()))

    x_, y_ = data.get_next(data="val")
    print(type(x_), type(x_.numpy()))
    print(x_.shape, x_.ndim)
    print(type(y_), type(y_.numpy()))
    print(y_.shape, y_.ndim)
    for i in range(x_.shape[0]):
        show_image_cv(x_.numpy()[i,...], title="img-val")
        print(np.max(x_.numpy()))

    x_, y_ = data.get_next()
    print(type(x_), type(x_.numpy()))
    print(x_.shape, x_.ndim)
    print(type(y_), type(y_.numpy()))
    print(y_.shape, y_.ndim)
    for i in range(x_.shape[0]):
        show_image_cv(x_.numpy()[i,...], title="img-train")
        print(np.max(x_.numpy()))

