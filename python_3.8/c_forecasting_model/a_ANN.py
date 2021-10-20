from util.create_model import create_BiLSTM, create_GRU, create_ANN, create_DNN
from util.create_dataset import concat_data

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# set random seed
tf.random.set_seed(1)
random.seed(1)
np.random.seed(1)

print(tf.__version__)       # 2.3.0

random_sample_save_folder_path = '/Report_github_code/python_3.8/b_data_processing/b_data_sampling/sampled_data/'
for i in range(1, 6):
    df = pd.read_csv( random_sample_save_folder_path + 'RE_demand_sample%i.csv' %i, index_col=0)
    regions = df.columns

    result = pd.DataFrame(index=['rmse_train', 'r2_train', 'mae_train', 'rmse_test', 'r2_test', 'mae_test'])
    predict = pd.DataFrame()
    for region in regions:
        # data reshape
        RE_demand = pd.read_csv(random_sample_save_folder_path + 'RE_demand_sample%i.csv' % i, index_col=0)   # data initialization
        RE_demand = RE_demand[region]
        RE_demand = pd.DataFrame(RE_demand)
        n_in = 24   # n_in : Past data step to be used for forecasting
        n_out = 1   # n_out : Future step to be forecast
        RE_demand=concat_data(RE_demand, n_in=n_in, n_out=n_out)

        # Separate the independent variable x and the dependent variable y
        x = RE_demand [ :, :n_in]
        y = RE_demand [ :, n_in:]

        # data scale
        x_scaler = RobustScaler()    # Robust scaler : Minimize the influence of outliers using median and IQR.
        y_scaler = RobustScaler()

        x_scaled = x_scaler.fit_transform(x)
        y_scaled = y_scaler.fit_transform(y)

        # train_test_split
        train_test_split = int(len(RE_demand)*0.8)

        scaled_train_x = x_scaled[:train_test_split, :]
        scaled_test_x = x_scaled[train_test_split:, :]

        scaled_train_y = y_scaled[:train_test_split, :]
        scaled_test_y = y_scaled[train_test_split:, :]

        # reshape the dataset to forecasting model input shape
        scaled_train_x = scaled_train_x.reshape(-1, n_in, 1)
        scaled_test_x = scaled_test_x.reshape(-1, n_in, 1)

        scaled_train_y = scaled_train_y.reshape(-1, 1)
        scaled_test_y = scaled_test_y.reshape(-1, 1)

        # Select Model & Checkpoint callbacks
        # Create a callback to save weight
        select_model = 'ANN'
        checkpoint_path = './%s/region/%s/weight/' %(select_model, region)
        checkpoint_file_path = '%s/sample%i_cp-{epoch:04d}.ckpt' % (checkpoint_path, i)
        checkpoint_dir = os.path.dirname(checkpoint_file_path)
        cp_callback = ModelCheckpoint(filepath=checkpoint_file_path, monitor='val_loss', save_weights_only=True,
                                          save_best_only=True, verbose=1)

        # Create Model
        nUnit = 60              #  Number of unit of hidden layer

        if select_model == 'BiLSTM':
            model = create_BiLSTM(nUnit, n_in)
        elif select_model == 'GRU':
            model = create_GRU(nUnit, n_in)
        elif select_model == 'DNN':
            model = create_DNN(nUnit, n_in)
        else:
            model = create_ANN(nUnit, n_in)

        model.save_weights(checkpoint_file_path.format(epoch=0))

        # Setting epoch & batch size
        model.fit(scaled_train_x, scaled_train_y, epochs=100, batch_size=6, validation_split=0.2,
                      callbacks=[cp_callback], verbose=1)
        model.summary()

        # load the last checkpoint
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        if select_model == 'BiLSTM':
            model = create_BiLSTM(nUnit, n_in)
        elif select_model == 'GRU':
            model = create_GRU(nUnit, n_in)
        elif select_model == 'DNN':
            model = create_DNN(nUnit, n_in)
        else:
            model = create_ANN(nUnit, n_in)

        model.load_weights(latest)
        model.summary()

        # Evaluation
        y_train = y_scaler.inverse_transform(scaled_train_y)
        y_test = y_scaler.inverse_transform(scaled_test_y)

        y_train_pr = model.predict(scaled_train_x).reshape(-1, 1)
        y_train_pr = y_scaler.inverse_transform(y_train_pr)

        y_test_pr = model.predict(scaled_test_x)
        y_test_pr = model.predict(scaled_test_x).reshape(-1,1)

        forecast_data = y_test_pr.copy()

        y_test_pr = y_scaler.inverse_transform(y_test_pr)

        # performance evaluation
        rmse_tr = np.sqrt(mean_squared_error(y_train, y_train_pr))
        r2s_tr = r2_score(y_train, y_train_pr)
        mae_tr = mean_absolute_error(y_train, y_train_pr)

        rmse_te = np.sqrt(mean_squared_error(y_test, y_test_pr))
        r2s_te = r2_score(y_test, y_test_pr)
        mae_te = mean_absolute_error(y_test, y_test_pr)

        print('-' * 30)


        metrics = [rmse_tr, r2s_tr, mae_tr, rmse_te, r2s_te, mae_te]
        result['%s' %region] = metrics
        performance_path = './%s/performance/' %(select_model)



        # forecasting
        # iteration 24 times one hour forecast to forecast one day
        forecast_data = forecast_data[-24:]
        forecast_data = np.concatenate(forecast_data)
        forecast_data = forecast_data.reshape(-1, n_in, 1)


        forecast = []

        for j in range(24):  # 24hour forecast
            fore = model.predict(forecast_data)
            forecast.append(fore)
            forecast_data = np.append([forecast_data], [fore])
            forecast_data = forecast_data[-24:]
            forecast_data = forecast_data.reshape(-1, n_in, 1)
        forecast = np.array(forecast)
        forecast = np.concatenate(forecast)
        forecast = y_scaler.inverse_transform(forecast)

        # data concatenate
        train = np.array(['train']).reshape(-1, 1)
        test = np.array(['test']).reshape(-1, 1)
        pred = np.array(['forecast']).reshape(-1, 1)

        forecast = np.concatenate([train, y_train_pr, test, y_test_pr, pred, forecast])
        forecast = np.concatenate(forecast)
        predict['%s' %region] = forecast
        forecast_path = './%s/forecast/' % (select_model)



    if not os.path.exists(performance_path):
        os.makedirs(performance_path)
    result.to_csv(performance_path + 'ANN_sample%s_score.csv' % i)

    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    predict.to_csv(forecast_path + 'ANN_sample%s_forecast.csv' % i)
