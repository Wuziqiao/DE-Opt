import argparse
import sys
import os
from tensorflow.python.keras.backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.filterwarnings("ignore")

from load_data_rating import *


if __name__ == '__main__':



    train_data, validation_data, test_data, n_user, n_item = load_data_rating(path='/home/sunbo/DE_Framwork/data/ml100k/100k.dat',
                                                             header=['user_id', 'item_id', 'rating', 't'],
                                                             test_size=0.2, sep='\t')