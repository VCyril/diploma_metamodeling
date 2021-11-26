import matplotlib.pyplot as plt
import numpy as np
import pyDOE as doe
import sys

import finite_elem as fin
import mlp

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

VAR_RANGE_COEF = 0.05
VAR_UPPER = 1 + VAR_RANGE_COEF
VAR_LOWER = 1 - VAR_RANGE_COEF

SAMPLES_NUM = 2


def create_samples(samples_num):
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

    n = len(var_upper)
    lhd = doe.lhs(n, samples=samples_num)
    lhd = lhd * (var_upper - var_lower) + var_lower

    return lhd


def run_simulation(_ir, _or, _r_groove, _r_neck, _t_shoulder, _h_inner, _h_shoulder, _h_groove, _h_neck, _h,
                   load):
    model, part = fin.create_sensor_geometry(_ir, _or, _r_groove, _r_neck, _t_shoulder,
                                             _h_inner, _h_shoulder, _h_groove, _h_neck, _h)
    fin.plot_model_geometry(model, False)
    fin.set_loads_and_constraints(model, part, load / (3.1415 * _r_neck ** 2))
    fin.set_mat_props(model, part)

    error = fin.set_elem_props(model, part, 'quad', 1)  # set 0.2
    if error != None:
        print(error)
        sys.exit()

    # fin.plot_elems_pressures_constraints(model, False)

    f = plt.figure()

    ax = f.add_subplot()

    rx, er, et = fin.solve_problem(model, _ir, _h_inner)
    # ax.scatter(rx, er, color='black')
    # ax.scatter(rx, et, color='blue')
    #
    # plt.show()

    n = len(rx)
    # result = {'ir': [_ir] * n,
    #           'or': [_or] * n,
    #           'r_groove': [_r_groove] * n,
    #           'r_neck': [_r_neck] * n,
    #           't_shoulder': [_t_shoulder] * n,
    #           'h_inner': [_h_inner] * n,
    #           'h_shoulder': [_h_shoulder] * n,
    #           'h_groove': [_h_groove] * n,
    #           'h_neck': [_h_neck] * n,
    #           'h': [_h] * n,
    #           'rx': rx,
    #           'er': er,
    #           'et': et}

    result = {'ir': _ir,
              'or': _or,
              'r_groove': _r_groove,
              'r_neck': _r_neck,
              't_shoulder': _t_shoulder,
              'h_inner': _h_inner,
              'h_shoulder': _h_shoulder,
              'h_groove': _h_groove,
              'h_neck': _h_neck,
              'h': _h,
              'rx': rx,
              'er': er,
              'et': et}

    return result


rated_load = [10000, 20000, 30000, 40000, 50000]

geom_samples = create_samples(SAMPLES_NUM)

# i = 0
# result = run_simulation(samples[i][0], samples[i][1], samples[i][2], samples[i][3],
#                         samples[i][4], samples[i][5], samples[i][6], samples[i][7],
#                         samples[i][8], samples[i][9], rated_load[0])
sim_results = []

for i in range(SAMPLES_NUM):
    res = run_simulation(geom_samples[i][0], geom_samples[i][1], geom_samples[i][2], geom_samples[i][3],
                         geom_samples[i][4], geom_samples[i][5], geom_samples[i][6], geom_samples[i][7],
                         geom_samples[i][8], geom_samples[i][9], rated_load[0])
    sim_results.append(res)

mlp.create_model_and_fit(sim_results)
