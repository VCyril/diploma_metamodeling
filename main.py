import matplotlib.pyplot as plt
import numpy as np
import pyDOE as doe
import pandas as pd
import pickle

import finite_elem as fin
import metamodeling as meta
import optimization as opt

# Задание геометрических параметров датчика
IR = 40
OR_ = 50
R_GROOVE = 20
R_NECK = 10
T_SHOULDER = 15
H_INNER = 27
H_SHOULDER = 44
H_GROOVE = 30
H_NECK = 40
H = 50

VAR_RANGE_COEF = 0.1
VAR_UPPER = 1 + VAR_RANGE_COEF
VAR_LOWER = 1 - VAR_RANGE_COEF

var_lower = np.array([IR * VAR_LOWER,
                      OR_ * VAR_LOWER,
                      R_GROOVE * VAR_LOWER,
                      R_NECK * VAR_LOWER,
                      T_SHOULDER * VAR_LOWER,
                      H_INNER * VAR_LOWER,
                      H_SHOULDER * VAR_LOWER,
                      H_GROOVE * VAR_LOWER,
                      H_NECK * VAR_LOWER,
                      H * VAR_LOWER])

var_upper = np.array([IR * VAR_UPPER,
                      OR_ * VAR_UPPER,
                      R_GROOVE * VAR_UPPER,
                      R_NECK * VAR_UPPER,
                      T_SHOULDER * VAR_UPPER,
                      H_INNER * VAR_UPPER,
                      H_SHOULDER * VAR_UPPER,
                      H_GROOVE * VAR_UPPER,
                      H_NECK * VAR_UPPER,
                      H * VAR_UPPER])

LOW_SAMPLES_NUM = 100
HIGH_SAMPLES_NUM = 20
LOW_MESH_SIZE = 2
HIGH_MESH_SIZE = 0.1
LOAD = 10000
SIM_RESULT_FNAME = 'sim_results_new.csv'
LOW_SIM_RES_FNAME = 'low_sim_results.csv'
HIGH_SIM_RES_FNAME = 'high_sim_results.csv'

x_cols = ['ir', 'or', 'r_groove', 'r_neck', 't_shoulder', 'h_inner',
          'h_shoulder', 'h_groove', 'h_neck', 'h', 'rx']


def create_samples(samples_num):
    n = len(var_upper)
    lhd = doe.lhs(n, samples=samples_num)
    lhd = lhd * (var_upper - var_lower) + var_lower

    return lhd


def create_mapping_samples(fname, samples_num, rx_num, low_model, high_model):
    geom_samples = create_samples(samples_num)
    with open(fname, 'w') as f:
        pass

    for i in range(samples_num):
        result = {'ir': [geom_samples[i][0]] * rx_num,
                  'or': [geom_samples[i][1]] * rx_num,
                  'r_groove': [geom_samples[i][2]] * rx_num,
                  'r_neck': [geom_samples[i][3]] * rx_num,
                  't_shoulder': [geom_samples[i][4]] * rx_num,
                  'h_inner': [geom_samples[i][5]] * rx_num,
                  'h_shoulder': [geom_samples[i][6]] * rx_num,
                  'h_groove': [geom_samples[i][7]] * rx_num,
                  'h_neck': [geom_samples[i][8]] * rx_num,
                  'h': [geom_samples[i][9]] * rx_num,
                  'rx': np.linspace(0, 1, rx_num),
                  }
        df = pd.DataFrame(result)
        x = df.to_numpy()
        high_pred = high_model.predict(x)
        low_pred = low_model.predict(x)
        df['high_pred'] = high_pred
        df['low_pred'] = low_pred
        mapp = high_pred / low_pred
        df['mapp'] = mapp

        if i == 0:
            df.to_csv(fname, mode='a', index=False)
        else:
            df.to_csv(fname, mode='a', header=False, index=False)


def run_simulation(_ir, _or, _r_groove, _r_neck, _t_shoulder, _h_inner, _h_shoulder, _h_groove, _h_neck, _h,
                   load, mesh_size):
    model, part, error = fin.create_sensor_geometry(_ir, _or, _r_groove, _r_neck, _t_shoulder,
                                                    _h_inner, _h_shoulder, _h_groove, _h_neck, _h)
    if error is not None:
        print(error)
        return None

    # fin.plot_model_geometry(model, False)
    error = fin.set_loads_and_constraints(model, part, load / (3.1415 * _r_neck ** 2))
    if error is not None:
        print(error)
        return None

    fin.set_mat_props(model, part)

    error = fin.set_elem_props(model, 'quad', mesh_size)
    if error is not None:
        print(error)
        return None
    fin.plot_elems_pressures_constraints(model, False)
    rx, er, _, sx_max = fin.solve_problem(model, _ir, _h_inner)
    n = len(rx)
    result = {'ir': [_ir] * n,
              'or': [_or] * n,
              'r_groove': [_r_groove] * n,
              'r_neck': [_r_neck] * n,
              't_shoulder': [_t_shoulder] * n,
              'h_inner': [_h_inner] * n,
              'h_shoulder': [_h_shoulder] * n,
              'h_groove': [_h_groove] * n,
              'h_neck': [_h_neck] * n,
              'h': [_h] * n,
              'rx': rx,
              'er': er,
              'sx_max': [sx_max] * n}

    return result


def create_models():
    low_model = meta.create_model_and_fit(LOW_SIM_RES_FNAME)
    high_model = 1
    # high_model = meta.create_model_and_fit(HIGH_SIM_RES_FNAME)
    return low_model, high_model


def simulate(res_fname, mesh_size, samples_num):
    geom_samples = create_samples(samples_num)
    with open(res_fname, 'w') as f:
        pass

    for i in range(samples_num):
        res = run_simulation(geom_samples[i][0], geom_samples[i][1], geom_samples[i][2], geom_samples[i][3],
                             geom_samples[i][4], geom_samples[i][5], geom_samples[i][6], geom_samples[i][7],
                             geom_samples[i][8], geom_samples[i][9], LOAD, mesh_size)
        if res is None:
            continue
        df = pd.DataFrame(res)
        if i == 0:
            df.to_csv(res_fname, mode='a', index=False)
        else:
            df.to_csv(res_fname, mode='a', header=False, index=False)


def open_models():
    with open('saved_models' + "/{}_{}_{}.pickle".format('CatBoost', 'low', '0'), 'rb') as f:
        low_model = pickle.load(f)
    with open('saved_models' + "/{}_{}_{}.pickle".format('CatBoost', 'high', '0'), 'rb') as f:
        high_model = pickle.load(f)
    with open('saved_models' + "/{}_{}_{}.pickle".format('CatBoost', 'mapp', '2'), 'rb') as f:
        mapp_model = pickle.load(f)
    return low_model, high_model, mapp_model

# def create_mapping_model(fname, samples_num):
#     geom_samples = create_samples(samples_num)
#     with open(fname, 'w') as f:
#         pass


# simulate(LOW_SIM_RES_FNAME, LOW_MESH_SIZE, LOW_SAMPLES_NUM)
# simulate(HIGH_SIM_RES_FNAME, HIGH_MESH_SIZE, HIGH_SAMPLES_NUM)

# low_model = meta.create_model_and_fit(LOW_SIM_RES_FNAME, 'low')
# high_model = meta.create_model_and_fit(HIGH_SIM_RES_FNAME, 'high')
low_model, high_model, mapp_model = open_models()
#
create_mapping_samples("mapping_results1.csv", 1, 100, low_model, high_model)
# map_mod = meta.create_model_and_fit('mapping_results.csv', 'mapp')
# low_model, high_model = create_models()

# er_model = meta.create_model_and_fit(SIM_RESULT_FNAME)
# opt_var_lower = np.append(var_lower, np.array([0, 0]))
# opt_var_upper = np.append(var_upper, np.array([1, 1]))
#
# opt.optimize(opt_var_lower, opt_var_upper, er_model, None)
