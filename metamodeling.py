import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pickle
from tensorflow import keras
import model_constr as mod

RANDOM_STATE = 42
MLP_DROPOUT = 0.2
NUM_FOLDS = 5
N_TRIALS = 3
# model_categories = ['MLP', 'Polynomial', 'CatBoost', 'XGBoost', 'SVR']
model_categories = ['CatBoost']
x_cols = ['ir', 'or', 'r_groove', 'r_neck', 't_shoulder', 'h_inner',
          'h_shoulder', 'h_groove', 'h_neck', 'h', 'rx']
# er_col = 'er'
er_col = 'mapp'
# sx_col = 'sx_max'
sx_col = 'low_pred'
models_folder = 'saved_models'


def loss_relative(y_true, y_pred):
    diff = np.abs(y_true - y_pred) / (max(y_true) - min(y_true)) * 100
    return np.mean(diff)


def read_from_csv(res_fname):
    df = pd.read_csv(res_fname)
    x = df[x_cols].to_numpy()
    er = df[er_col].to_numpy()
    sx_max = df[sx_col].to_numpy()
    return x, er, sx_max


def objective(trial, X, Y, y_name):
    model_category = trial.suggest_categorical('model_category', model_categories)

    if model_category == 'MLP':
        n_layers_hidden = trial.suggest_int('n_layers_hidden', 1, 7)
        n_neurons_input = trial.suggest_int('n_neurons_input', 10, 50)
        n_neurons_hidden = trial.suggest_int('n_neurons_hidden', 10, 200)
        activation = trial.suggest_categorical('activation', ['relu', 'elu'])
        model = mod.make_MLP({'n_layers_hidden': n_layers_hidden,
                              'n_neurons_input': n_neurons_input,
                              'n_neurons_hidden': n_neurons_hidden,
                              'activation': activation})

    if model_category == 'Polynomial':
        degree = trial.suggest_int('degree', 2, 3)
        model = mod.make_Polynomial({'degree': degree})

    if model_category == 'CatBoost':
        max_depth = trial.suggest_int('max_depth', 6, 9)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        model = mod.make_CatBoost({'max_depth': max_depth,
                                   'learning_rate': learning_rate,
                                   'n_estimators': n_estimators})

    if model_category == 'XGBoost':
        max_depth = trial.suggest_int('max_depth', 3, 10)
        eta = trial.suggest_float('eta', 0.02, 0.5)
        n_estimators = trial.suggest_int('n_estimators', 100, 400)
        model = mod.make_XGBoost({'max_depth': max_depth,
                                  'eta': eta,
                                  'n_estimators': n_estimators})

    if model_category == 'SVR':
        C = trial.suggest_float('C', 0.1, 100, log=True)
        gamma = trial.suggest_float('gamma', 1e-4, 10, log=True)
        model = mod.make_SVR({'C': C,
                              'gamma': gamma})

    if model_category == 'MLP':
        kf = KFold(n_splits=NUM_FOLDS)
        errors = []
        for train_idx, test_idx in kf.split(X, Y):
            # model.fit(X[train_idx, :], Y[train_idx], epochs=50, verbose=0, validation_split=0.25)
            model.fit(X[train_idx, :], Y[train_idx])
            Y_pred = model.predict(X[test_idx, :])
            errors.append(mean_absolute_error(Y[test_idx], Y_pred))
        score = -np.array(errors)
    else:
        if model_category == 'CatBoost':
            model.fit(X, Y, logging_level='Silent')
        model.fit(X, Y)
        score = cross_val_score(model, X, Y, n_jobs=4, cv=5, scoring='neg_mean_absolute_error')

    if model_category == 'MLP':
        model.save(models_folder + "/{}_{}_{}.h5".format(model_category, y_name, trial.number))
    else:
        with open(models_folder + "/{}_{}_{}.pickle".format(model_category, y_name, trial.number), 'wb') as out:
            pickle.dump(model, out)

    accuracy = score.mean()
    return accuracy


def get_best_parameters(X, Y, y_name):
    study = optuna.create_study(direction='maximize')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
    f = lambda trial: objective(trial, x_train, y_train, y_name)
    study.optimize(f, n_trials=N_TRIALS)
    # pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    # complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    # print("Study statistics: ")
    # print("  Number of finished trials: ", len(study.trials))
    # print("  Number of pruned trials: ", len(pruned_trials))
    # print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    # print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return trial.params, trial.number


def create_model_and_fit(res_fname, model_name):
    x, er, sx_max = read_from_csv(res_fname)
    trial_results, trial_number = get_best_parameters(x, er, model_name)
    # model = getattr(mod, 'make_' + trial_results['model_category'])(trial_results)

    x_train, x_test, y_train, y_test = train_test_split(x, er, test_size=0.2, random_state=RANDOM_STATE)
    # kf = KFold(n_splits=NUM_FOLDS)
    # errors = []
    # for train_idx, test_idx in kf.split(x_train, y_train):
    #     model.fit(x_train[train_idx, :], y_train[train_idx])
    #     y_pred = model.predict(x_train[test_idx, :])
    #     errors.append(mean_absolute_error(y_train[test_idx], y_pred))
    # score = -np.array(errors)
    # # print(score)
    # score = score.mean()
    # print("Rated error for the model = {}".format(score))

    if trial_results['model_category'] == 'MLP':
        er_model = keras.models.load_model(models_folder + "/{}_{}_{}.h5".format(trial_results['model_category'],
                                                                                 er_col, trial_number))
    else:
        with open(models_folder + "/{}_{}_{}.pickle".format(trial_results['model_category'],
                                                            er_col, trial_number), 'rb') as f:
            er_model = pickle.load(f)
    y_pred = er_model.predict(x_test)
    print("Loss relative er = {}".format(loss_relative(y_test, y_pred)))

    # trial_results, trial_number = get_best_parameters(x, sx_max, sx_col)
    # x_train, x_test, y_train, y_test = train_test_split(x, sx_max, test_size=0.2, random_state=RANDOM_STATE)
    # if trial_results['model_category'] == 'MLP':
    #     sx_model = keras.models.load_model(models_folder + "/{}_{}_{}.h5".format(trial_results['model_category'],
    #                                                                              sx_col, trial_number))
    # else:
    #     with open(models_folder + "/{}_{}_{}.pickle".format(trial_results['model_category'],
    #                                                         sx_col, trial_number), 'rb') as f:
    #         sx_model = pickle.load(f)
    # y_pred = sx_model.predict(x_test)
    # print("Loss relative sx = {}".format(loss_relative(y_test, y_pred)))

    # return er_model, sx_model
    return er_model