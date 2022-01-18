import argparse
import tensorflow.compat.v1 as tf

import sys
import os
# import os.path
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from model.neumf import NeuMF
from model.DE_neumf import DE_NeuMF
from model.lrml import LRML
from model.DE_lrml import DE_LRML
from load_data_ranking import *


def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['NeuMF', 'LRML', 'DE_NeuMF', 'DE_LRML'],
                        default='DE_NeuMF')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    batch_size = args.batch_size

    train_data, validation_data, test_data, n_user, n_item = load_data_neg(path='/home/sunbo/DE_Framwork/data/ml1m/ratings.dat', test_size=0.2, sep="::")

    # for i in range(5):
    #     print('-------------------------------------------------------------------')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True



    with tf.Session(config=config) as sess:
        model = None

        if args.model == "LRML":
            model = LRML(sess, n_user, n_item)
        if args.model == "DE_LRML":
            model = DE_LRML(sess, n_user, n_item)
        if args.model == "NeuMF":
            model = NeuMF(sess, n_user, n_item, )
        if args.model == "DE_NeuMF":
            model = DE_NeuMF(sess, n_user, n_item, lr_min=0, lr_max=0.1, reg_rate_min=0, reg_rate_max=0.01)
        # build and execute the model
        if model is not None:
            model.build_network()
            model.execute(train_data, validation_data, test_data)
