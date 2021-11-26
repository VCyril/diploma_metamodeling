import tensorflow as tf
from tensorflow import keras
import numpy as np

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


def transform_sim_results(samples):
    X = []
    Y = []
    for sample in samples:
        const_params = np.array([[sample['ir'], sample['or'], sample['r_groove'],
                                  sample['r_neck'], sample['t_shoulder'], sample['h_inner'],
                                  sample['h_shoulder'], sample['h_groove'], sample['h_neck'],
                                  sample['h']]] * len(sample['rx']))

        rx = np.array(sample['rx'])

        x = np.concatenate((const_params, rx[:, None]), axis=1)
        for row in x:
            X.append(row)
        y = np.array(sample['er'])
        for val in y:
            Y.append(val)
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y


def create_model_and_fit(samples):
    model = keras.Sequential()
    model.add(keras.layers.Dense(20, input_shape=(11,)))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(20, activation='elu'))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    x, y = transform_sim_results(samples)

    model.fit(x, y, epochs=100, verbose=1)
