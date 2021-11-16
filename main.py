import parsing as pars
import plotting as myplt
import matplotlib.pyplot as plt
import finite_elem as fin
import numpy as np
import pyDOE as doe
import converting_results as conv
import sys

SAMPLES_NUM = 10

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

ir = np.arange(IR * VAR_LOWER, IR * VAR_UPPER, IR * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
or_ = np.arange(OR_ * VAR_LOWER, OR_ * VAR_UPPER, OR_ * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
r_groove = np.arange(R_GROOVE * VAR_LOWER, R_GROOVE * VAR_UPPER, R_GROOVE * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
r_neck = np.arange(R_NECK * VAR_LOWER, R_NECK * VAR_UPPER, R_NECK * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
t_shoulder = np.arange(T_SHOULDER * VAR_LOWER, T_SHOULDER * VAR_UPPER, T_SHOULDER * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
h_inner = np.arange(H_INNER * VAR_LOWER, H_INNER * VAR_UPPER, H_INNER * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
h_shoulder = np.arange(H_SHOULDER * VAR_LOWER, H_SHOULDER * VAR_UPPER, H_SHOULDER * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
h_groove = np.arange(H_GROOVE * VAR_LOWER, H_GROOVE * VAR_UPPER, H_GROOVE * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
h_neck = np.arange(H_NECK * VAR_LOWER, H_NECK * VAR_UPPER, H_NECK * 2 * VAR_RANGE_COEF / SAMPLES_NUM)
h = np.arange(H * VAR_LOWER, H * VAR_UPPER, H * 2 * VAR_RANGE_COEF / SAMPLES_NUM)


def create_samples():
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
    lhd = doe.lhs(n, samples=SAMPLES_NUM)
    lhd = lhd * (var_upper - var_lower) + var_lower
    # samples = [[var_lower[ind] + (var_upper[ind] - var_lower[ind]) * lhd_point[ind]
    #             for ind in range(n)] for lhd_point in lhd]
    return lhd


def run_simulation(_ir, _or, _r_groove, _r_neck, _t_shoulder, _h_inner, _h_shoulder, _h_groove, _h_neck, _h,
                   load):
    model, part = fin.create_sensor_geometry(_ir, _or, _r_groove, _r_neck, _t_shoulder,
                                             _h_inner, _h_shoulder, _h_groove, _h_neck, _h)
    fin.plot_model_geometry(model, False)
    fin.set_loads_and_constraints(model, part, load / (3.1415 * _r_neck ** 2))
    fin.set_mat_props(model, part)

    error = fin.set_elem_props(model, part, 'quad', 1)
    if error != None:
        print(error)
        sys.exit()

    # fin.plot_elems_pressures_constraints(model, False)

    f = plt.figure()

    ax = f.add_subplot()

    rx, er, et = fin.solve_problem(model, _ir, _h_inner)
    ax.scatter(rx, er, color='black')
    plt.show()


rated_load = [10000, 20000, 30000, 40000, 50000]

samples = create_samples()

# print(samples[0])

for i in range(len(samples)):
    run_simulation(samples[i][0], samples[i][1], samples[i][2], samples[i][3],
                   samples[i][4], samples[i][5], samples[i][6], samples[i][7],
                   samples[i][8], samples[i][9], rated_load[0])
