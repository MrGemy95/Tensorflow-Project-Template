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
        #self.is_training = tf.placeholder(tf.bool)

        model = importlib.import_module(self.config.arch)

        self.net        = model.gen_net()

        self.optimizer  = model.gen_optimizer()

        self.accuracy   = (lambda logits, labels: model.compute_acc(logits, labels))

        self.loss       = (lambda logits, labels: model.compute_loss(logits, labels))

        self.summary    = (lambda loss, accuracy: model.setup_summary(loss, accuracy))

        ######## for prediction
        #self.pred       = model.predict(logits)
        #self.metrics, self.metrics_update = model.metrics_eval(self.pred, self.y)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        #self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        # https://www.tensorflow.org/guide/checkpoint
        self.checkpoint = tf.train.Checkpoint(step=self.cur_epoch_tensor,
                                              optimizer=self.optimizer,
                                              model=self.net)
        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                  directory=self.config.checkpoint_dir,
                                                  max_to_keep=self.config.max_to_keep,
                                                  keep_checkpoint_every_n_hours=None,
                                                  checkpoint_name="ckpt")

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

