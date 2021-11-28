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

LOAD = 10000

VAR_RANGE_COEF = 0.2
VAR_UPPER = 1 + VAR_RANGE_COEF
VAR_LOWER = 1 - VAR_RANGE_COEF

SAMPLES_NUM = 20


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
    model, part, error = fin.create_sensor_geometry(_ir, _or, _r_groove, _r_neck, _t_shoulder,
                                                    _h_inner, _h_shoulder, _h_groove, _h_neck, _h)
    if error is not None:
        print(error)
        return None

    fin.plot_model_geometry(model, False)
    error = fin.set_loads_and_constraints(model, part, load / (3.1415 * _r_neck ** 2))
    if error is not None:
        print(error)
        return None

    fin.set_mat_props(model, part)

    error = fin.set_elem_props(model, part, 'quad', 1)  # set 0.2
    if error is not None:
        print(error)
        return None

    # fin.plot_elems_pressures_constraints(model, False)

    f = plt.figure()

    ax = f.add_subplot()

    rx, er, et = fin.solve_problem(model, _ir, _h_inner)
    # ax.scatter(rx, er, color='black')
    # ax.scatter(rx, et, color='blue')
    #
    # plt.show()

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


geom_samples = create_samples(SAMPLES_NUM)

sim_results = []

for i in range(SAMPLES_NUM):
    res = run_simulation(geom_samples[i][0], geom_samples[i][1], geom_samples[i][2], geom_samples[i][3],
                         geom_samples[i][4], geom_samples[i][5], geom_samples[i][6], geom_samples[i][7],
                         geom_samples[i][8], geom_samples[i][9], LOAD)
    if res is None:
        continue
    sim_results.append(res)

print("Simulation results:")
print(sim_results)
# mlp.create_model_and_fit(sim_results)
