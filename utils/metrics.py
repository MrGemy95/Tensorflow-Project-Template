"""
This file will contain the metrics of the framework
"""
import tensorflow as tf


def top_k_error(labels, predictions, k, name=None):
    with tf.name_scope(name, default_name="top_k_error"):
        labels = tf.expand_dims(tf.to_int32(labels), axis=-1)
        _, top_k = tf.nn.top_k(predictions, k=k)
        in_top_k = tf.reduce_mean(tf.to_float(tf.equal(top_k, labels)), -1)
        return 1 - in_top_k


class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg


class AverageMeterList:
    def __init__(self, num_cls):
        self.cls = num_cls
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls
        self.reset()

    def reset(self):
        self.value = [0] * self.cls
        self.avg = [0] * self.cls
        self.sum = [0] * self.cls
        self.count = [0] * self.cls

    def update(self, val, n=1):
        for i in range(self.cls):
            self.value[i] = val[i]
            self.sum[i] += val[i] * n
            self.count[i] += n
            self.avg[i] = self.sum[i] / self.count[i]

    @property
    def val(self):
        return self.avg


class FPSMeter:
    """
    Class to measure frame per second in our networks
    """

    def __init__(self, batch_size):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0
        self.milliseconds = 0.0

        self.batch_size = batch_size

    def reset(self):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0

    def update(self, seconds):
        self.milliseconds += seconds * 1000
        self.frame_count += self.batch_size

        self.frame_per_second = self.frame_count / (self.milliseconds / 1000.0)
        self.f_in_milliseconds = self.milliseconds / self.frame_count

    @property
    def mspf(self):
        return self.f_in_milliseconds

    @property
    def fps(self):
        return self.frame_per_second

    def print_statistics(self):
        print("""
Statistics of the FPSMeter
Frame per second: {:.2f} fps
Milliseconds per frame: {:.2f} ms in one frame
These statistics are calculated based on
{:d} Frames and the whole taken time is {:.4f} Seconds
        """.format(self.frame_per_second, self.f_in_milliseconds, self.frame_count, self.milliseconds / 1000.0))
