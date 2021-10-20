from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    return dataframe.values[1:, 1:]


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/RES/'

    filenames = list()
    # RE demand/solar
    filenames += ['total_2030_label.csv', 'jeju_2030_label.csv']
    # load input data
    X = load_group(filenames, filepath)
    return X


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    total_data = load_dataset_group('train', prefix + 'RES/')
    return total_data

total_data = load_dataset()