# cnn model
from numpy import mean
from numpy import std
import numpy as np
from numpy import dstack
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from load_data import load_dataset
total_data = load_dataset()
import random
import tensorflow as tf
import os
from models.VAE_CONV_D1 import VariationalAutoencoder

# set random seed
np.random.seed(1)
random.seed(1)
tf.random.set_random_seed(1)

print(tf.__version__)        # 1.13.2


# data scaling
scaler = MinMaxScaler(feature_range=(0, 1))
total_data = scaler.fit_transform(total_data.reshape(total_data.shape[0], -1)).reshape(total_data.shape)



# asign data path
SECTION = 'vae_conv_d1'
RUN_ID = '0001'
DATA_NAME = 'res'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE = 'build'

# set VAE using CONV_1D as neron (z_dim is compressor parameter)
VAE = VariationalAutoencoder(
    input_dim = (26,2)
    , encoder_conv_filters = [32,64,64,64]
    , encoder_conv_kernel_size = [3,3,3,3]
    , encoder_conv_strides = [1,1,2,1]
    , decoder_conv_t_filters = [64,64,32,1]
    , decoder_conv_t_kernel_size = [3,3,3,3]
    , decoder_conv_t_strides=[1,1,2,1]
    , z_dim = 2
)

if MODE == 'build':
    VAE.save(RUN_FOLDER)
else:
    VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# set hyper-parameter
LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000

BATCH_SIZE = 32
EPOCHS = 500
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

VAE.compile(LEARNING_RATE, R_LOSS_FACTOR)

VAE.train(
    total_data
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)

