from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from keras.layers import Dropout
from keras.optimizers import adam_v2
import catboost as cb
import xgboost as xgb


def make_MLP(params):
    model = keras.Sequential()
    model.add(Dropout(0.1, input_shape=(11,)))
    model.add(keras.layers.Dense(params['n_neurons_input'], input_shape=(11,)))
    for i in range(params['n_layers_hidden']):
        model.add(Dropout(0.1))
        model.add(keras.layers.Dense(params['n_neurons_hidden'], activation=params['activation']))
    model.add(Dropout(0.1))
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mae')
    return model


def make_XGBoost(params):
    model = xgb.XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                             eta=params['eta'], subsample=0.7, colsample_bytree=0.8)
    return model


def make_Polynomial(params):
    model = make_pipeline(PolynomialFeatures(params['degree']), LinearRegression())
    return model


def make_CatBoost(params):
    model = cb.CatBoostRegressor(loss_function='MAE', learning_rate=params['learning_rate'],
                                 depth=params['max_depth'], l2_leaf_reg=0.1, n_estimators=params['n_estimators'])
    return model


def make_SVR(params):
    model = SVR(C=params['C'], gamma=params['gamma'])
    return model
