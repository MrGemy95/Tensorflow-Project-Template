import os,sys
import importlib
parentdir = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0,parentdir)

from models.net_model import NetModel
from trainers.net_trainer import NetTrainer
from trainers.net_predictor import NetPredictor
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config, args.train)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = None
    # create your data generator
    dm = importlib.import_module(config.dataset)
    data = dm.DataSet(config)
    # create an instance of the model you want
    model = NetModel(config, data)
    # create tensorboard logger
    logger = Logger(sess, config)

    # here you train your model
    if (args.train):
        # create trainer and pass all the previous components to it
        trainer = NetTrainer(sess, model, data, config, logger)
        # load model if exists
        model.load(sess)
        trainer.train()
    else:
        # create predictor and pass all the previous components to it
        predictor = NetPredictor(sess, model, data, config, logger)
        # load model if exists
        model.load(sess)
        predictor.predict()


if __name__ == '__main__':
    main()
