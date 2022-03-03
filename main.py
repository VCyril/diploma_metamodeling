import matplotlib.pyplot as plt
import numpy as np
import pyDOE as doe
import pandas as pd
import sys

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

SAMPLES_NUM = 100
ELEM_SIZE = 0.2
LOAD = 10000
SIM_RESULT_FNAME = 'sim_results_new.csv'


def create_samples(samples_num):
    n = len(var_upper)
    lhd = doe.lhs(n, samples=samples_num)
    lhd = lhd * (var_upper - var_lower) + var_lower

    return lhd


def run_simulation(_ir, _or, _r_groove, _r_neck, _t_shoulder, _h_inner, _h_shoulder, _h_groove, _h_neck, _h,
                   load):
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

    error = fin.set_elem_props(model, 'quad', ELEM_SIZE)
    if error is not None:
        print(error)
        return None

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


geom_samples = create_samples(SAMPLES_NUM)

sim_results = []

# with open(SIM_RESULT_FNAME, 'w') as f:
#     pass
#
# for i in range(SAMPLES_NUM):
#     res = run_simulation(geom_samples[i][0], geom_samples[i][1], geom_samples[i][2], geom_samples[i][3],
#                          geom_samples[i][4], geom_samples[i][5], geom_samples[i][6], geom_samples[i][7],
#                          geom_samples[i][8], geom_samples[i][9], LOAD)
#     if res is None:
#         continue
#     df = pd.DataFrame(res)
#     if i == 0:
#         df.to_csv(SIM_RESULT_FNAME, mode='a', index=False)
#     else:
#         df.to_csv(SIM_RESULT_FNAME, mode='a', header=False, index=False)

# er_model, sx_model = meta.create_model_and_fit(SIM_RESULT_FNAME)
er_model = meta.create_model_and_fit(SIM_RESULT_FNAME)
opt_var_lower = np.append(var_lower, np.array([0, 0]))
opt_var_upper = np.append(var_upper, np.array([1, 1]))
# opt.optimize(opt_var_lower, opt_var_upper, er_model, sx_model)

opt.optimize(opt_var_lower, opt_var_upper, er_model, None)
