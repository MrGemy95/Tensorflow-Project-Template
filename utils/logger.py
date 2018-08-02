import os

import tensorflow as tf


class DefinedSummarizer:
    def __init__(self, sess, summary_dir, scalar_tags=None, images_tags=None):
        """
        :param sess: The Graph tensorflow session used in your graph.
        :param summary_dir: the directory which will save the summaries of the graph
        :param scalar_tags: The tags of summaries you will use in your training loop
        :param images_tags: The tags of image summaries you will use in your training loop
        """
        self.sess = sess

        self.scalar_tags = scalar_tags
        self.images_tags = images_tags

        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.init_summary_ops()

        self.summary_writer = tf.summary.FileWriter(summary_dir)

        if "comet_api_key" in config:
            from comet_ml import Experiment
            self.experiment = Experiment(api_key=config['comet_api_key'], project_name=config['exp_name'])
            self.experiment.log_multiple_params(config)



    def set_summaries(self, scalar_tags=None, images_tags=None):
        self.scalar_tags = scalar_tags
        self.images_tags = images_tags
        self.init_summary_ops()

    def init_summary_ops(self):
        with tf.variable_scope('summary_ops'):
            if self.scalar_tags is not None:
                for tag in self.scalar_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            if self.images_tags is not None:
                for tag, shape in self.images_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                    self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

    def summarize(self, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param step: the number of iteration in your training
        :param summaries_dict: the dictionary which contains your summaries .
        :param summaries_merged: Merged summaries which they come from your graph
        :return:
        """
        if summaries_dict is not None:
            summary_list = self.sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_writer.add_summary(summary, step)
        if summaries_merged is not None:
            self.summary_writer.add_summary(summaries_merged, step)

            if hasattr(self, 'experiment') and self.experiment is not None:
                self.experiment.log_multiple_metrics(summaries_dict, step=step)


    def finalize(self):
        self.summary_writer.flush()


class Logger:
    def __init__(self, sess,config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

        if "comet_api_key" in config:
            from comet_ml import Experiment
            self.experiment = Experiment(api_key=config['comet_api_key'], project_name=config['exp_name'])
            self.experiment.disable_mp()
            self.experiment.log_multiple_params(config)




    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
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
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]),
                                                                            name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)

                if hasattr(self,'experiment') and self.experiment is not None:
                    self.experiment.log_multiple_metrics(summaries_dict, step=step)

                summary_writer.flush()