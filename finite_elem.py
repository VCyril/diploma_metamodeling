import pycalculix as pyc
import pycalculix.results_file
import numpy as np

ccx = '/usr/bin/ccx'
cgx = '/usr/bin/cgx'
gmsh = '/home/cyril/.local/lib/python3.8/site-packages/gmsh-4.8.4-Linux64-sdk/bin/gmsh'
model_name = 'model'


def get_nodal_strain(self, fname='', display=True, levels=21, gradient=False,
                     gmult=1.0, max_val=None, min_val=None, title=''):
    sel = {}
    sel['nodes'] = self.__problem.fea.view.nodes
    sel['elements'] = self.__problem.fea.view.elements
    sel['faces'] = self.__problem.fea.view.faces

    # sort nodes low to high so index is correct
    # we have index to id below so showing subsets works
    sel['nodes'] = list(sel['nodes'])
    sel['nodes'] = sorted(sel['nodes'], key=lambda k: k.id)
    # store results at nodes

    rx = []
    yx = []
    er = []
    et = []

    axials = []

    id_to_ind = {}
    for node in sel['nodes']:
        id_to_ind[node.id] = len(axials)
        rx.append(node.x)
        yx.append(node.y)
        axials.append(1)
        er.append(self.__results[self.__time]['node'][node.id]['ex'])
        et.append(self.__results[self.__time]['node'][node.id]['ez'])

    return rx, yx, er, et


def create_sensor_geometry():
    # We'll be modeling a rotating jet engine part
    model = pyc.FeaModel(model_name, ccx, cgx, gmsh)
    model.set_units('mm')
    show_gui = True

    # радиальные размеры
    ir = 35  # внутренний размер мембраны
    rf = 15
    or_ = 50  # внешний радиус датчика в целом
    r_groove = 20  # радиус центра внешней проточки
    r_neck = 8.3  # радиус шеи датчика
    r_head = 12.5  # радиус головы датчика
    t_shoulder = 14.36

    # осевые размеры
    h_inner = 27  # высота внутренней полости
    h_shoulder = 44  # высота плеч датчика
    h_groove = 29.4  # высота  угла внешней проточки
    h_neck = 41.1  # высота шеи
    h = 50  # высота датчика в целом

    m50_loop = [
        [ir, 0],
        [0, -h_inner],
        [rf, 0],
        [0, h_shoulder],
        [-t_shoulder, 0],
        [-(or_ - t_shoulder - r_groove), h_groove - h_shoulder],
        [(r_neck - r_groove), (h_neck - h_groove)],
        [(r_head - r_neck), 5],
        [0, 5],
        [-r_head, 0],
        [0, -(h_neck + 10 - h_inner)]
    ]

    part = pyc.Part(model)
    part.goto(0, h_inner)
    lines = []
    for [drad, dax] in m50_loop:
        [L, p1, p2] = part.draw_line_delta(drad, dax)
        lines.append(L)

    fillet_list = [[0, 1, 3], [5, 6, 3], [6, 7, 3]]
    for [i1, i2, rad] in fillet_list:
        part.fillet_lines(lines[i1], lines[i2], rad)

    return model, part


def set_loads_and_constraints(model, part):
    model.set_load('press', part.right, 63661977)
    model.set_constr('fix', part.bottom, 'x')
    model.set_constr('fix', part.left, 'y')


def plot_model_geometry(model, show_gui):
    model.plot_geometry(model_name + '_prechunk_areas', lnum=False,
                        pnum=False, display=show_gui)
    model.plot_geometry(model_name + '_prechunk_lines', pnum=False,
                        display=show_gui)
    model.plot_geometry(model_name + '_prechunk_points', lnum=False,
                        display=show_gui)


def set_mat_props(model, part):
    mat = model.make_matl('steel')
    mat.set_mech_props(7800, 210 * (10 ** 9), 0.3)
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


def solve_problem(model):
    prob = pyc.Problem(model, 'struct', 'problem')
    prob.solve()
    # disp = True
    # fields = 'S2'
    # fields = fields.split(',')
    # for field in fields:
    #     fname = model_name + '_' + field
    #     prob.rfile.nplot(field, fname, display=disp)
    #
    # model.view.print_summary()

    rx, yx, er, et = prob.rfile.get_nodal_strain()

    # _, _, ur, _ = prob.rfile.get_nodal_strain()

    rx = np.array(rx)
    yx = np.array(yx)
    er = np.array(er)
    et = np.array(et)

    # plt.scatter(rx, yx)

    delta_r = 0.01e-3
    delta_a = 0.01e-3

    ir = 35
    h_inner = 27

    id_ = (rx <= ir) & (np.abs((yx - h_inner)) < delta_a)

    id_2 = yx < delta_a

    r_foot = rx[id_2]
    y_foot = yx[id_2]

    # plt.scatter(r_foot, ur[id_2])

    rx = rx[id_]
    yx = yx[id_]
    er = er[id_]
    et = et[id_]

    return rx, er, et
