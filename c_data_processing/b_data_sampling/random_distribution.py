import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

# set random seed
np.random.seed(1)
random.seed(1)

# Decide the day range of the data
# Empty date is removed from the weekend filter
# Select a sample for each date using filter_index.
start = datetime(2030,1,3)
end = datetime(2030,12,13)
filter_index = pd.period_range(start=start, end=end, freq='D')
filter_index = pd.DataFrame(index=filter_index)
filter_index.reset_index(inplace=True)
filter_index = filter_index.astype('str')
filter_index = filter_index.values
filter_index = np.concatenate(filter_index)

# data frame index
start = datetime(2030,1,3)
end = datetime(2030,12,14)
df_index = pd.period_range(start=start, end=end, freq='h')
df_index = df_index[:-1]
df_index.name = 'date'

# data load
# df = pd.read_csv('a')
df = pd.read_csv('../a_weekend_label_filter/labeled_data/after_weekend_filter.csv', index_col=0)
print(df.head())
result = []

# 13 region * 5 sample
column_name = ['seoul', 'busan', 'daegu', 'incheon', 'gwangju', 'daejeon', 'ulsan', 'sejong', 'gyeonggi', 'gangwon',
               'chungbuk', 'chungnam', 'jeonbuk', 'jeonnam', 'gyeongbuk', 'gyeongnam', 'jeju']
for i in range(1):
    for date in range(len(filter_index)):
        z = []
        # If the number of populations is greater than the number of samples, duplicate extraction is not allowed.
        if len(df.loc[filter_index[date]]) > 85:
            population = df.loc[filter_index[date]]
            population = population.values
            population = list(population)
            sample = random.sample(population, 85)
            sample = np.array(sample)
            sample = sample.transpose()
            result.append(sample)

        # If the number of populations is less than the number of samples, duplicate extraction is allowed.
        else:
            for iteration in range(85):
                population = df.loc[filter_index[date]]
                population = population.values
                population = list(population)
                sample = random.choice(population)
                z.append(sample)
            z=np.array(z)
            z=z.transpose()
            result.append(z)

    result = np.array(result)
    result = np.concatenate(result)

sample_1 = []
sample_2 = []
sample_3 = []
sample_4 = []
sample_5 = []

# Selected sample distribution

a=0     # number of region
b=17    # number of region

random_sample_save_folder_path = './sampled_data/'

for iteration in range(1,6):
    globals()['sample_{}'.format(iteration)].append(result[:,a:b])
    globals()['sample_{}'.format(iteration)] = np.concatenate(globals()['sample_{}'.format(iteration)])
    globals()['sample_{}'.format(iteration)] = pd.DataFrame(globals()['sample_{}'.format(iteration)], index=df_index, columns = column_name)


    if not os.path.exists(random_sample_save_folder_path):
        os.makedirs(random_sample_save_folder_path)

    globals()['sample_{}'.format(iteration)].to_csv(random_sample_save_folder_path + 'power_demand_sample%i.csv' %iteration)
    a += 17
    b += 17

