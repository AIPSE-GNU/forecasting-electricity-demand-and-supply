from keras.models import Model
from keras.layers import Dense, Flatten, Input, LSTM, GRU, LeakyReLU, Bidirectional
from keras.optimizers import Adam

# ANN
def create_ANN(units, step):

    Inputs = Input(batch_shape=(None, step, 1))

    x = Dense(units, activation=LeakyReLU(alpha=0.1))(Inputs)
    x = Flatten()(x)

    Outputs = Dense(1)(x)

    model = Model(Inputs, Outputs)

    model.compile(loss='mse', optimizer=Adam(lr=0.01))

    return model

# DNN
def create_DNN(units, step):

    Inputs = Input(batch_shape=(None, step, 1))

    x = Dense(units//3, activation=LeakyReLU(alpha=0.1))(Inputs)
    x = Dense(units//3, activation=LeakyReLU(alpha=0.1))(x)
    x = Dense(units//3, activation=LeakyReLU(alpha=0.1))(x)

    x = Flatten()(x)

    Outputs = Dense(1)(x)

    model = Model(Inputs, Outputs)

    model.compile(loss='mse', optimizer=Adam(lr=0.01))

    return model

# GRU
def create_GRU(units, step):

    Inputs = Input(batch_shape=(None, step, 1))
    x = GRU(units, activation=LeakyReLU(alpha=0.1))(Inputs)
    Outputs = Dense(1)(x)

    model = Model(Inputs, Outputs)

    model.compile(loss='mse', optimizer=Adam(lr=0.01))

    return model

# LSTM
def create_BiLSTM(units, step):

    Inputs = Input(batch_shape=(None, step, 1))
    x = Bidirectional(LSTM(units))(Inputs)
    Outputs = Dense(1)(x)

    model = Model(Inputs, Outputs)

    model.compile(loss='mse', optimizer='adam')

    return model
