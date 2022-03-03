import numpy as np
import pandas as pd
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import os

N_VAR = 12
N_OBJ = 3


class SensorEvaluation(Problem):

    def __init__(self, var_lower, var_upper, er_model, sx_model):
        super().__init__(n_var=N_VAR, n_obj=N_OBJ, xl=var_lower, xu=var_upper)
        self.er_model = er_model
        self.sx_model = sx_model

    def get_stress(self, x):
        ln = len(x)
        sx_x = np.array([x[:ln - 1]])
        sx = self.sx_model.predict(sx_x)
        # print(sx[0])
        return sx[0]

    def get_strain(self, x):
        ln = len(x)
        e1_x = np.array([x[:ln - 1]])
        e1 = self.er_model.predict(e1_x)
        e2_x = np.array([np.append(x[:ln - 2], x[ln - 1])])
        e2 = self.er_model.predict(e2_x)

        x1 = []
        for i in range(50):
            x1.append(np.array([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], 0.02 * i]))
        # print(np.array(x1))
        e_full = self.er_model.predict(np.array(x1))
        er_max = max((max(e_full)), abs(min(e_full)))

        return e1[0], e2[0], er_max

    def _evaluate(self, x, out, *args, **kwargs):
        M = x.shape[0]
        gfs = []
        for i in range(M):
            # sigma = self.get_stress(x[i, :])
            e1, e2, er_max = self.get_strain(x[i, :])
            sigma = er_max * 2e5
            rkp = -np.abs((e1 - e2) * 1000)
            gfs.append([sigma, rkp, np.abs(np.abs(e1 / e2) - 1)])

        out["F"] = np.array(gfs)
        # out["G"] = -(np.array(skos) - self.thresh/2)


def optimize(var_lower, var_upper, er_model, sx_model):
    problem = SensorEvaluation(var_lower, var_upper, er_model, sx_model)

    method = get_algorithm("nsga2",
                           pop_size=20,
                           sampling=get_sampling("real_random"),
                           crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
                           mutation=get_mutation("real_pm", eta=3.0),
                           eliminate_duplicates=True,
                           )

    res = minimize(problem,
                   method,
                   termination=('n_gen', 500),
                   seed=1,
                   save_history=True,
                   verbose=True)

    print(res.X, res.F)

    plt.rcParams.update({'font.size': 20})

    f = plt.figure(figsize=(12, 12))
    ax = f.add_subplot(111, projection="3d")
    ax.scatter(res.F[:, 0], -res.F[:, 1], res.F[:, 2])
    ax.set_xlabel(r'$\sigma_{max}$', fontsize=25, labelpad=20)
    ax.set_ylabel('РКП', fontsize=25, labelpad=20)
    ax.set_zlabel(r'$|\frac{\varepsilon_1}{\varepsilon_2} - 1|$', fontsize=25, labelpad=20)

    plt.show()
