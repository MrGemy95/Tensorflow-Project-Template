import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        if self.config.tf_version[0] < 2:
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                              self.sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))
        else:
            self.train_summary_writer = tf.summary.create_file_writer(os.path.join(self.config.summary_dir, "train"))
            self.test_summary_writer = tf.summary.create_file_writer(os.path.join(self.config.summary_dir, "test"))

        self.summarize_funcion = {
            "1":self.summarize_v1,
            "2":self.summarize_v2
        }

    # it can summarize scalars and images.
    def summarize_v1(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()

    # for tensorflow2.2
    def summarize_v2(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        summary_writer = self.train_summary_writer if summarizer == "train" else self.test_summary_writer
        with tf.name_scope(scope):
            if summaries_dict is not None:
                #https://www.tensorflow.org/tensorboard/get_started
                with summary_writer.as_default():
                    for tag, value in summaries_dict.items():
                        if len(value.shape) <= 1:
                            tf.summary.scalar(tag, float(value), step = step)
                        else:
                            tf.summary.image(tag, value, step = step)

    def summarize(self, *args, **kwargs):
        self.summarize_funcion[str(self.config.tf_version[0])](*args, **kwargs)