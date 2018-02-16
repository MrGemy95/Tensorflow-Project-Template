import pickle

import tensorflow as tf


class Cifar100DataLoaderNumpy:
    """
    This will contain the dataset api
    It will load the numpy files from the pkl file which is dumped by prepare_cifar100.py script
    Please make sure that you have included all of the needed config
    Thanks..
    """

    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

        with open(self.config.data_numpy_pkl, "rb") as f:
            self.data_pkl = pickle.load(f)

        self.x_train = self.data_pkl['x_train']
        self.y_train = self.data_pkl['y_train']
        self.x_test = self.data_pkl['x_test']
        self.y_test = self.data_pkl['y_test']

        print('x_train: ', self.x_train.shape, self.x_train.dtype)
        print('y_train: ', self.y_train.shape, self.y_train.dtype)
        print('x_test: ', self.x_test.shape, self.x_test.dtype)
        print('y_test: ', self.y_test.shape, self.y_test.dtype)

        self.train_len = self.x_train.shape[0]
        self.test_len = self.x_test.shape[0]

        print("Data loaded successfully..")

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self.build_dataset_api()

    def build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.float32, [None] + list(self.x_train.shape[1:]))
            self.labels_placeholder = tf.placeholder(tf.int32, [None, ])

            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            self.dataset = self.dataset.batch(self.config.batch_size)

            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                            self.dataset.output_shapes)

            self.init_iterator_op = self.iterator.make_initializer(self.dataset)

            self.next_batch = self.iterator.get_next()

            print("X_batch shape dtype: ", self.next_batch[0].shape)
            print("Y_batch shape dtype: ", self.next_batch[1].shape)

    def initialize_train(self):
        self.sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_train,
                                                        self.labels_placeholder: self.y_train})

    def initialize_test(self):
        self.sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_test,
                                                        self.labels_placeholder: self.y_test})


def main():
    class Config:
        data_numpy_pkl = "../data/cifar-100-python/data_numpy.pkl"
        data_mode = "numpy"

        image_height = 32
        image_width = 32
        batch_size = 8

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = Cifar100DataLoaderNumpy(sess, Config)

    x, y = data_loader.next_batch

    data_loader.initialize_train()

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize_test()

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)


if __name__ == '__main__':
    main()
