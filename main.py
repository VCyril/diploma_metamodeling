import parsing as pars
import plotting as myplt
import matplotlib.pyplot as plt
import finite_elem as fin
import numpy as np
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

ir = np.arange(IR * 0.95, IR * 1.05, IR * 0.1 / SAMPLES_NUM)
or_ = np.arange(OR_ * 0.95, OR_ * 1.05, OR_ * 0.1 / SAMPLES_NUM)
r_groove = np.arange(R_GROOVE * 0.95, R_GROOVE * 1.05, R_GROOVE * 0.1 / SAMPLES_NUM)
r_neck = np.arange(R_NECK * 0.95, R_NECK * 1.05, R_NECK * 0.1 / SAMPLES_NUM)
t_shoulder = np.arange(T_SHOULDER * 0.95, T_SHOULDER * 1.05, T_SHOULDER * 0.1 / SAMPLES_NUM)
h_inner = np.arange(H_INNER * 0.95, H_INNER * 1.05, H_INNER * 0.1 / SAMPLES_NUM)
h_shoulder = np.arange(H_SHOULDER * 0.95, H_SHOULDER * 1.05, H_SHOULDER * 0.1 / SAMPLES_NUM)
h_groove = np.arange(H_GROOVE * 0.95, H_GROOVE * 1.05, H_GROOVE * 0.1 / SAMPLES_NUM)
h_neck = np.arange(H_NECK * 0.95, H_NECK * 1.05, H_NECK * 0.1 / SAMPLES_NUM)
h = np.arange(H * 0.95, H * 1.05, H * 0.1 / SAMPLES_NUM)


def run_simulation(_ir, _or, _r_groove, _r_neck, _t_shoulder, _h_inner, _h_shoulder, _h_groove, _h_neck, _h,
                   load):
    model, part = fin.create_sensor_geometry(_ir, _or, _r_groove, _r_neck, _t_shoulder,
                                             _h_inner, _h_shoulder, _h_groove, _h_neck, _h)
    # fin.plot_model_geometry(model, False)
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

for i in range(10):
    run_simulation(ir[i], or_[i], r_groove[i], r_neck[i], t_shoulder[i],
                   h_inner[i], h_shoulder[i], h_groove[i], h_neck[i], h[i], rated_load[0])
