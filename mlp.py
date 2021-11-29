import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics
import optuna
import pandas as pd

# loss_tracker = keras.metrics.Mean(name="loss")
# mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

columns_x = ['ir', 'or', 'r_groove', 'r_neck', 't_shoulder', 'h_inner',
             'h_shoulder', 'h_groove', 'h_neck', 'h', 'rx']


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


def read_from_csv(res_fname):
    df = pd.read_csv(res_fname)
    x = df[columns_x].to_numpy()
    y = df['er'].to_numpy()
    print(x)
    print(y)
    return x, y


def get_best_parameters(X, Y):
    def objective(trial):
        n_layers_hidden = trial.suggest_int('n_layers_hidden', 1, 7)
        n_neurons_input = trial.suggest_int('n_neurons_input', 10, 50)
        n_neurons_hidden = trial.suggest_int('n_neurons_hidden', 10, 200)
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        model = keras.Sequential()
        model.add(keras.layers.Dense(n_neurons_input, input_shape=(11,)))
        for i in range(n_layers_hidden):
            model.add(keras.layers.Dense(n_neurons_hidden, activation=activation))
        model.add(keras.layers.Dense(1, activation='linear'))

        model.compile(optimizer='adam', loss='mae')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        error = sklearn.metrics.mean_absolute_error(y_pred, y_test)

        return error / (np.max(y_test) - np.min(y_test)) * 100

    study = optuna.create_study()
    study.optimize(objective, n_trials=25)
    # pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial.params


def create_model_and_fit(res_fname):
    # x, y = transform_sim_results(samples)
    x, y = read_from_csv(res_fname)
    trial_results = get_best_parameters(x, y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(trial_results['n_neurons_input'], input_shape=(11,)))
    for i in range(trial_results['n_layers_hidden']):
        model.add(keras.layers.Dense(trial_results['n_neurons_hidden'], activation=trial_results['activation']))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(x_train, y_train, epochs=25, verbose=0)
    y_pred = model.predict(x_test)

    error = sklearn.metrics.mean_squared_error(y_pred, y_test)
    print("Rated error for the model = {}".format(error))
