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

def inference(inputs, phase_train):
    config = g_config.get_cfg()
    with tf.variable_scope(config.exp_name):
        # Conv => ReLu => Pool
        x = Conv2D(filters=6, kernel_size=5, padding="same", activation=None, name="block1_conv1")(inputs)
        x = tf.keras.layers.BatchNormalization(name="batch_norm1")(x, training = phase_train)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)
        # Conv => ReLu => Pool input: 14*14*6
        x = Conv2D(filters=16, kernel_size=5, padding="same", activation=None, name='block1_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name="batch_norm2")(x, training = phase_train)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_poo2')(x)
        # Conv => ReLu => Pool input: 7*7*16
        # flat
        x = Flatten(name='flatten')(x)
        # Dense
        x = Dense(units=128, activation="relu", name="f1")(x)
        x = Dense(units=84, activation="relu", name="f2")(x)
        # softmax分类器
        logits = Dense(units=config.num_class, name="prediction")(x)
    return logits
    #reference:
    #https://blog.csdn.net/zaf0516/article/details/89958962
    #https://www.machinecurve.com/index.php/2020/01/15/how-to-use-batch-normalization-with-keras/


def loss(logits, labels, ignore_label=-1, cb=None, name='loss'):
    with tf.name_scope(name):
        #prob = tf.nn.softmax(logits)
        loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
        loss = tf.reduce_mean(loss)
    return loss


def acc(logits, labels, ignore_label=-1, name='acc'):
    with tf.name_scope(name):
        prob = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def predict(logits, name='predict'):
    with tf.name_scope(name):
        prob = tf.squeeze(tf.nn.softmax(logits))
        pred = tf.squeeze(tf.cast(tf.argmax(prob, axis=-1), tf.int32))
    return pred

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

##################### None Tensorflow part  #####################
def post_proc(x_raw, y_raw, predict):
    """
    do some post data process in test procedure, here should be finished WITHOUT tensorflow API
    :param x_raw:
    :param y_raw:
    :param predict:
    :return:
    """
    print("y_raw:{} predict:{}".format(y_raw, predict))