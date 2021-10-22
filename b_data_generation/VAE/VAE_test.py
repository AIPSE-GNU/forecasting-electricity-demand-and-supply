import numpy as np
from util.loaders import load_model
from models.VAE_CONV_D1 import VariationalAutoencoder
import pandas as pd
import tensorflow as tf
import random

np.random.seed(1)
random.seed(1)
tf.random.set_random_seed(1)

# asign dataset path
SECTION = 'vae_conv_d1'
RUN_ID = '0001'
DATA_NAME = 'res'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

# load a model with extracted features from ex_VAE_002_train.py
VAE = load_model(VariationalAutoencoder, RUN_FOLDER)


from sklearn.preprocessing import MinMaxScaler
from load_data import load_dataset

# down scale through MinMaxScaler
total_data = load_dataset()
scaler = MinMaxScaler(feature_range=(0, 1))
total_data = scaler.fit_transform(total_data.reshape(total_data.shape[0], -1)).reshape(total_data.shape)

# set sample size
n_to_show = 365*1000
znew = np.random.normal(size=(n_to_show, VAE.z_dim))
reconst_test = VAE.decoder.predict(np.array(znew))

# up scale generated sample through MinMaxScaler
x_test = scaler.inverse_transform(reconst_test.reshape(reconst_test.shape[0], -1)).reshape(reconst_test.shape)

# separate dataset and save each sample, separated dataset
x_test_total = x_test[:,:,0]
x_test_jeju = x_test[:,:,1]

generative_test_demand = pd.DataFrame(x_test_total)
generative_test_demand.to_csv("./RES/vae_generative_test_total_sample.csv", header=[i for i in range(1,27,1)])
generative_test_solar = pd.DataFrame(x_test_jeju)
generative_test_solar.to_csv("./RES/vae_generative_test_jeju_sample.csv", header=[i for i in range(1,27,1)])
