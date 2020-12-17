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
        # ======== for training ========
        self.optimizer  = model.gen_optimizer()

        self.accuracy   = (lambda logits, labels: model.compute_acc(logits, labels))

        self.loss       = (lambda logits, labels: model.compute_loss(logits, labels))

        self.summary    = (lambda loss, accuracy: model.setup_summary(loss, accuracy))
        # ======== for prediction ========
        self.pred       = (lambda logits: model.compute_predict(logits))
        # post processing
        self.pred_proc  = (lambda logits, x_raw, y_raw: model.compute_predict_proc(logits, x_raw, y_raw))
        #self.metrics, self.metrics_update = model.metrics_eval(self.pred, self.y)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        # https://www.tensorflow.org/guide/checkpoint
        self.checkpoint = tf.train.Checkpoint(
            step=self.cur_epoch_tensor,
            optimizer=self.optimizer,
            model=self.net)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.config.checkpoint_dir,
            max_to_keep=self.config.max_to_keep,
            keep_checkpoint_every_n_hours=None,
            checkpoint_name="ckpt")

    # save function that saves the checkpoint in the path defined in the config file
    def save_v2(self, sess = None):
        interval = 1 if int(self.config.num_save_interval) <= 0 else int(self.config.num_save_interval)
        if not self.config.is_train:
            print("Saving model...")
            self.net.save(self.config.saved_model_dir)
            print("==== PB Model saved ===")
            self.net.save(self.config.checkpoint_dir + "../model.h5",  save_format='h5')
            print("==== h5 Model saved ====")
        elif int(self.checkpoint.step) % interval == 0:
            print("Saving model...")
            self.manager.save()
            print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load_v2(self, sess = None):
        latest_checkpoint = self.manager.latest_checkpoint
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.checkpoint.restore(latest_checkpoint)
            print("Model loaded")