from util.create_dataset import concat_data
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import os


# set random seed
random.seed(1)
tf.random.set_seed(1)
np.random.seed(1)

# parameter
# C : the value increases, margin error increases
# degree : order of model
# coef0 : How much the model is influenced by the order
# kernel : rbf, poly, linear, sigmoid
# epsilon : range of margin

# load data
random_sample_save_folder_path = '/Report_github_code/python_3.8/b_data_processing/b_data_sampling/sampled_data/'


for i in range(1, 6):
    df = pd.read_csv(random_sample_save_folder_path + 'RE_demand_sample%i.csv' % i, index_col=0)
    regions = df.columns

    result = pd.DataFrame(index=['rmse_train', 'r2_train', 'mae_train', 'rmse_test', 'r2_test', 'mae_test'])
    predict = pd.DataFrame()

    for region in regions:
        # data reshape
        RE_demand = pd.read_csv(random_sample_save_folder_path + 'RE_demand_sample%i.csv' % i, index_col=0)  # data initialization
        RE_demand = RE_demand[region]
        RE_demand = pd.DataFrame(RE_demand)
        n_in = 24  # n_in : Past data step to be used for forecasting
        n_out = 1  # n_out : Future step to be forecast
        RE_demand = concat_data(RE_demand, n_in=n_in, n_out=n_out)

        # Separate the independent variable x and the dependent variable y
        x = RE_demand[:, :n_in]
        y = RE_demand[:, n_in:]

        # data scale
        x_scaler = RobustScaler()  # Robust scaler : Minimize the influence of outliers using median and IQR.
        y_scaler = RobustScaler()

        x_scaled = x_scaler.fit_transform(x)
        y_scaled = y_scaler.fit_transform(y)

        # train_test_split
        train_test_split = int(len(RE_demand) * 0.8)

        scaled_train_x = x_scaled[:train_test_split, :]
        scaled_test_x = x_scaled[train_test_split:, :]

        scaled_train_y = y_scaled[:train_test_split, :]
        scaled_test_y = y_scaled[train_test_split:, :]

        # reshape the dataset to forecasting model input shape
        scaled_train_y = scaled_train_y.reshape(-1, 1)
        scaled_test_y = scaled_test_y.reshape(-1, 1)

        # model create
        reg_SVR = SVR(kernel='rbf')
        reg_SVR.fit(scaled_train_x, scaled_train_y)

        # model evaluation
        y_train = y_scaler.inverse_transform(scaled_train_y)
        y_test = y_scaler.inverse_transform(scaled_test_y)

        train_pred = reg_SVR.predict(scaled_train_x)
        test_pred = reg_SVR.predict(scaled_test_x)

        fore = test_pred.copy()

        train_pred = train_pred.reshape(-1,1)
        test_pred = test_pred.reshape(-1,1)

        train_pred = y_scaler.inverse_transform(train_pred)
        test_pred = y_scaler.inverse_transform(test_pred)

        train_rmse = sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)

        test_rmse = sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        metrics = [train_rmse, train_r2, train_mae, test_rmse, test_r2, test_mae]
        result['%s' %region] = metrics
        performance_path = './SVR/performance/'

        # data forecasting
        forecast_data = fore[-24:]
        forecast_data = forecast_data.reshape(1,-1)

        forecast = []

        for j in range(24):
            fore = reg_SVR.predict(forecast_data)
            forecast.append(fore)
            forecast_data = np.append([forecast_data], [forecast])
            forecast_data = forecast_data[-24:]
            forecast_data = forecast_data.reshape(1, -1)

        forecast = np.array(forecast).reshape(-1, 1)
        forecast = y_scaler.inverse_transform(forecast)

        # data concatenate
        train = np.array(['train']).reshape(-1, 1)
        test = np.array(['test']).reshape(-1, 1)
        pred = np.array(['forecast']).reshape(-1, 1)

        forecast = np.concatenate([train, train_pred, test, test_pred, pred, forecast])
        forecast = np.concatenate(forecast)
        predict['%s' %region] = forecast
        forecast_path = './SVR/forecast/'


    if not os.path.exists(performance_path):
        os.makedirs(performance_path)
    result.to_csv(performance_path + 'SVR_sample%s_score.csv' % i)

    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    predict.to_csv(forecast_path + 'SVR_sample%s_forecast.csv' % i)


