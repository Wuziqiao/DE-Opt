import argparse
import sys
import os
from tensorflow.python.keras.backend import set_session
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore")

from model.AutoRec import *
from model.DE_AutoRec import *
from model.nrr import *
from model.DE_nrr import *
from load_data_rating import *



def parse_args():
    parser = argparse.ArgumentParser(description='nnRec')
    parser.add_argument('--model', choices=['I-AutoRec', 'U-AutoRec', 'DE_I-AutoRec', 'DE_U-AutoRec', 'NRR', 'DE_NRR', 'AFM'],
                        default='I-AutoRec')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)  # 128 for unlimpair
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # 1e-4 for unlimpair
    parser.add_argument('--reg_rate', type=float, default=0.1)  # 0.01 for unlimpair
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--display_step', type=int, default=1000)
    parser.add_argument('--show_time', type=bool, default=False)
    parser.add_argument('--T', type=int, default=2)
    parser.add_argument('--deep_layers', type=str, default="200, 200, 200")
    parser.add_argument('--field_size', type=int, default=10)
    parser.add_argument('--lr_min', type=float, default=0.00001)
    parser.add_argument('--lr_max', type=float, default=0.01)
    parser.add_argument('--reg_rate_min', type=float, default=0.0001)
    parser.add_argument('--reg_rate_max', type=float, default=0.01)
    parser.add_argument('--total_round', type=int, default=-0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    reg_rate = args.reg_rate
    num_factors = args.num_factors
    display_step = args.display_step
    show_time = args.show_time
    lr_min = args.lr_min
    lr_max = args.lr_max
    reg_rate_min = args.reg_rate_min
    reg_rate_max = args.reg_rate_max
    total_round = args.total_round


    train_data, validation_data, test_data, n_user, n_item = load_data_rating(path='/home/sunbo/DE_Framwork/data/ml1m/ratings.dat',
                                                             header=['user_id', 'item_id', 'rating', 't'],
                                                             test_size=0.2, sep='::')
    # print('Epinion')
    import tensorflow.compat.v1 as tf

    for i in [0.000001]:
        for j in [10, 1, 0.1, 0.01, 0.001, 0]:
            learning_rate = i
            reg_rate = j

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                model = None
                # model selection
                if args.model == 'I-AutoRec':
                    model = IAutoRec(sess, n_user, n_item, learning_rate, reg_rate)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)
                if args.model == 'U-AutoRec':
                    model = UAutoRec(sess, n_user, n_item)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)
                if args.model == 'DE_I-AutoRec':
                    model = DE_IAutoRec(sess, n_user, n_item)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)
                if args.model == 'DE_U-AutoRec':
                    model = DE_UAutoRec(sess, n_user, n_item)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)
                if args.model == 'NRR':
                    model = NRR(sess, n_user, n_item, learning_rate, reg_rate)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)
                if args.model == 'DE_NRR':
                    model = DE_NRR(sess, n_user, n_item)
                    model.build_network()
                    model.execute(train_data, validation_data, test_data)




