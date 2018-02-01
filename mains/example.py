from utils.config import process_config
from utils.utils import get_args
from utils.dirs import create_dirs
from logger import Logger
from models.example_model import ExampleModel
from data_loader.data_generator import DataGenerator
from trainers.example_trainer import ExampleTrainer
import tensorflow as tf

def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])

    sess = tf.Session()
    model=ExampleModel(config)
    data=DataGenerator(config)
    logger=Logger(sess,config)
    trainer= ExampleTrainer(sess,model,data,config,logger)
    trainer.train()
if __name__ == '__main__':
    main()
