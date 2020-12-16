""" letnet.py
    Implementation of lenet5
    Reference:
    https://zhuanlan.zhihu.com/p/51714932
    https://blog.csdn.net/qq_27825451/article/details/102457051
    https://blog.csdn.net/weixin_41695564/article/details/80240106
    https://geektutu.com/post/tensorflow-mnist-simplest.html


    keras
    https://blog.csdn.net/u010472607/article/details/82319295
"""
import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Flatten, Dense
from tensorflow.keras import Model

import utils.config_val as g_config

#lenet5

def fetch_data():
    config = g_config.get_cfg()
    with tf.variable_scope("inputs"):
        # input image
        x      = tf.placeholder(tf.float32, shape=[config.batch_size, config.input_h, config.input_w, config.input_c])
        x_test = tf.placeholder(tf.float32, shape=[config.batch_size, config.input_h, config.input_w, config.input_c])
        # labels
        y      = tf.placeholder(tf.int32, shape=config.output_shape)
        y_test = tf.placeholder(tf.int32, shape=config.output_shape)
    return x, y, x_test, y_test


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(filters=6, kernel_size=5, padding="same", activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.conv2 = Conv2D(filters=16, kernel_size=5, padding="same", activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(84, activation='relu')
        self.d3 = Dense(10, activation=None)
        # https://www.freesion.com/article/9805814731/
        self._set_inputs(tf.TensorSpec([None, 28, 28, 1], tf.float32, name = "inputs"))  

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        o = self.d3(x)
        return o

# Create an instance of the model
def gen_net():
    net = MyModel()
    net.build(input_shape=(None, 28, 28, 1))
    net.summary()
    return net

def gen_optimizer():
    optimizer = tf.keras.optimizers.Adam()
    return optimizer

def compute_loss(logits, labels, ignore_label=-1, name='loss'):
    with tf.name_scope(name):
        #prob = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
    return loss

def compute_acc(logits, labels, ignore_label=-1, name='acc'):
    with tf.name_scope(name):
        prob = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def compute_predict(logits, name='predict'):
    with tf.name_scope(name):
        prob = tf.squeeze(tf.nn.softmax(logits))
        pred = tf.squeeze(tf.cast(tf.argmax(prob, axis=-1), tf.int32))
    return pred

def compute_predict_proc(logits, x_raw, y_raw):
    prob = tf.squeeze(tf.nn.softmax(logits))
    pred = tf.squeeze(tf.cast(tf.argmax(prob, axis=-1), tf.int32))
    gt   = tf.squeeze(tf.cast(tf.argmax(y_raw, axis=-1), tf.int32))

    print("pred:{}".format(pred))
    print("gt:  {}".format(gt))
    return None

def metrics_eval(predict, labels, name="metrics"):
    with tf.name_scope(name):
        y = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        metrics, metrics_op = tf.metrics.accuracy(labels=y, predictions=predict)
    return metrics, metrics_op

def train_op(loss, global_steps, opt_name, **kwargs):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = _get_optimizer(opt_name, kwargs)
        op = optimizer.minimize(loss, global_step=global_steps)
    return op


def setup_summary(loss, acc):
    #summary_loss = tf.summary.scalar('loss', loss)
    #summary_acc = tf.summary.scalar('acc', acc)
    #return tf.summary.merge([summary_loss, summary_acc])
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    return tf.summary.merge_all()

def _get_optimizer(opt_name, params):
    if opt_name == 'adam':
        return tf.train.AdamOptimizer(params['lr'])
    elif opt_name == 'adadelta':
        return tf.train.AdadeltaOptimizer(params['lr'])
    elif opt_name == 'sgd':
        return tf.train.GradientDescentOptimizer(params['lr'])
    elif opt_name == 'momentum':
        return tf.train.MomentumOptimizer(params['lr'], params['momentum'])
    elif opt_name == 'rms':
        return tf.train.RMSPropOptimizer(params['lr'])
    elif opt_name == 'adagrad':
        return tf.train.AdagradOptimizer(params['lr'])
    else:
        print('error')

#test
if __name__ == '__main__':

    class Train():
        def __init__(self):
            self.net = gen_net()

            self.optimizer = gen_optimizer()

            self.accuracy = (lambda logits, labels: compute_acc(logits, labels))

            self.loss = (lambda logits, labels: compute_loss(logits, labels))

        def train_step(self, model, optimizer, images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = self.loss(logits, labels)
            # compute gradient
            grads = tape.gradient(loss, model.trainable_variables)
            # update to weights
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            acc = self.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

        def test_step(self, model, images, labels):
            logits = model(images, training=False)
            loss = self.loss(logits, labels)
            acc = self.accuracy(logits, labels)
            # loss and accuracy is scalar tensor
            return logits, loss, acc

    train = Train()

    image = tf.random.normal(shape=[2,28,28,1], dtype=tf.float32)
    print(image.numpy().shape)
    label = tf.one_hot(tf.range(0, 2), depth=10, axis=-1)
    for i in range(20):
        _, loss, acc = train.train_step(train.net, train.optimizer, image, label)
        print("train loss:{}, acc:{}".format(loss.numpy(), acc.numpy()))

        if i%3 == 0:
            _, loss, acc = train.test_step(train.net, image, label)
            print("test loss:{}, acc:{}".format(loss.numpy(), acc.numpy()))
