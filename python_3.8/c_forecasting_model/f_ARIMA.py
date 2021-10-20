from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

import pandas as pd
import numpy as np
import os
import random

# set random seed
random.seed(1)
np.random.seed(1)

random_sample_save_folder_path = '/Report_github_code/python_3.8/b_data_processing/b_data_sampling/sampled_data/'
for i in range(1, 6):
    df = pd.read_csv( random_sample_save_folder_path + 'RE_demand_sample%i.csv' %i, index_col=0)
    regions = df.columns

    result = pd.DataFrame(index=['rmse_test', 'r2_test', 'mae_test'])
    predict = pd.DataFrame()

    for region in regions:
        RE_demand = pd.read_csv(random_sample_save_folder_path + 'RE_demand_sample%i.csv' % i, index_col=0)  # data initialization
        RE_demand = RE_demand[region]
        RE_demand = pd.DataFrame(RE_demand)


        # train_test_split
        train_test_split = int(len(RE_demand)*0.8)
        train, test = RE_demand[:train_test_split], RE_demand[train_test_split:]

        # data scaling
        scaler = RobustScaler()
        scaler = scaler.fit(RE_demand.values)

        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)


        # model setting
        history = [x for x in train_scaled]

        test_pred = []

        for j in range(len(test_scaled)):
            model = ARIMA(history, order=(3,1,1))   # setting (p, d, q) guide : https://www.youtube.com/watch?v=YQF5PDDI9jo&list=LL&index=5
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output
            test_pred.append(yhat)
            obs = test_scaled[i]
            history.append(obs)
        test_pred = np.array(test_pred)
        test_pred = scaler.inverse_transform(test_pred)

        # model evalutaion
        rmse = sqrt(mean_squared_error(test, test_pred))
        r2 = r2_score(test, test_pred)
        mae = mean_absolute_error(test, test_pred)

        metrics = [rmse, r2, mae]
        result['%s' %region] = metrics
        performance_path = './ARIMA/performance/'


        # data forecasting
        forecast = model_fit.forecast(steps=24)
        forecast = forecast.reshape(-1,1)
        forecast = scaler.inverse_transform(forecast)


        # data concatenate
        test = np.array(['test']).reshape(-1, 1)
        pred = np.array(['forecast']).reshape(-1, 1)

        forecast = np.concatenate([test, test_pred, pred, forecast])
        forecast = np.concatenate(forecast)
        predict['%s' % region] = forecast

        forecast_path = './ARIMA/forecast/'


    if not os.path.exists(performance_path):
        os.makedirs(performance_path)
    result.to_csv(performance_path + 'ARIMA_sample%s_score.csv' % i)

    if not os.path.exists(forecast_path):
        os.makedirs(forecast_path)
    predict.to_csv(forecast_path + 'ARIMA_sample%s_forecast.csv' % i)