import pycalculix as pyc
import pycalculix.results_file
import numpy as np

ccx = '/usr/bin/ccx'
cgx = '/usr/bin/cgx'
gmsh = '/home/cyril/.local/lib/python3.8/site-packages/gmsh-4.8.4-Linux64-sdk/bin/gmsh'
model_name = 'model'


def create_sensor_geometry(ir=40, or_=50, r_groove=20, r_neck=10, t_shoulder=15,
                           h_inner=27, h_shoulder=44, h_groove=30, h_neck=40, h=50):
    model = pyc.FeaModel(model_name, ccx, cgx, gmsh)
    model.set_units('mm')

    line_loop = [
        [ir, 0],
        [0, -h_inner],
        [or_ - ir, 0],
        [0, h_shoulder],
        [-t_shoulder, 0],
        [-(or_ - t_shoulder - r_groove), h_groove - h_shoulder],
        [(r_neck - r_groove), (h_neck - h_groove)],
        [0, h - h_neck],
        [-r_neck, 0],
        [0, -(h - h_inner)]
    ]

    part = pyc.Part(model)
    part.goto(0, h_inner)
    lines = []
    for [drad, dax] in line_loop:
        [L, p1, p2] = part.draw_line_delta(drad, dax)
        lines.append(L)

    fillet_list = [[0, 1, 3], [5, 6, 3], [6, 7, 3]]
    for [i1, i2, rad] in fillet_list:
        try:
            part.fillet_lines(lines[i1], lines[i2], rad)
        except Exception as excp:
            return None, None, str(excp)
    return model, part, None


def set_loads_and_constraints(model, part, load):
    try:
        model.set_load('press', part.right, load)
    except Exception as excp:
        return str(excp)

    try:
        model.set_constr('fix', part.bottom, 'x')
    except Exception as excp:
        return str(excp)

    try:
        model.set_constr('fix', part.left, 'y')
    except Exception as excp:
        return str(excp)
    return None


def plot_model_geometry(model, show_gui):
    model.plot_geometry(model_name + '_prechunk_areas', lnum=False,
                        pnum=False, display=show_gui)
    model.plot_geometry(model_name + '_prechunk_lines', pnum=False,
                        display=show_gui)
    model.plot_geometry(model_name + '_prechunk_points', lnum=False,
                        display=show_gui)


def set_mat_props(model, part):
    mat = model.make_matl('steel')
    mat.set_mech_props(7800, 200e5, 0.3)
    model.set_matl(mat, part)


def set_elem_props(model, part, elem_shape, elem_size):
    model.set_eshape(elem_shape, 2)
    model.set_etype('axisym', 'A0')
    try:
        model.mesh(elem_size, 'fineness', 'gmsh')
    except Exception as excp:
        return str(excp)
    return None


def plot_elems_pressures_constraints(model, show_gui):
    model.plot_elements(display=show_gui)
    model.plot_pressures('press')
    model.plot_constraints('constr')


def solve_problem(model, ir, h_inner):
    prob = pyc.Problem(model, 'struct', 'problem')
    prob.solve()

    rx, yx, er, et = prob.rfile.get_nodal_strain()

    rx = np.array(rx)
    yx = np.array(yx)
    er = np.array(er)
    et = np.array(et)

    delta_r = 0.01e-3
    delta_a = 0.01e-3

    id_ = (rx <= ir) & (np.abs((yx - h_inner)) < delta_a)

    rx = rx[id_]
    yx = yx[id_]
    er = er[id_]
    et = et[id_]

    rx = rx / ir

    return rx, er, et
