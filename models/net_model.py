from base.base_model import BaseModel
import importlib
import tensorflow as tf


class NetModel(BaseModel):
    def __init__(self, config, data=None):
        super(NetModel, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        model = importlib.import_module(self.config.arch)

        if (self.config.dataset_type == "TF") and (not self.data is None):
            self.x, self.y = self.data.get_next()
        else:
            self.x, self.y, _, _ = model.fetch_data()

        logits          = model.inference(self.x, self.is_training)

        self.accuracy   = model.acc(logits, self.y)

        self.loss       = model.loss(logits, self.y)

        self.summary    = model.setup_summary(self.loss, self.accuracy)

        #train_op
        self.train_step = model.train_op(self.loss, self.global_step_tensor, self.config.optimizer,
                                         lr=self.config.learning_rate, momentum=self.config.momentum)

        ######## for prediction
        self.pred       = model.predict(logits)
        self.metrics, self.metrics_update = model.metrics_eval(self.pred, self.y)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def post_proc(self, x_raw, y_raw, predict):
        """
        do some post data process in test procedure, here should be finished WITHOUT tensorflow API
        :param x_raw:
        :param y_raw:
        :param predict:
        :return:
        """
        model = importlib.import_module(self.config.arch)
        model.post_proc(x_raw, y_raw, predict)
