import sys
import pycalculix as pyc

ccx = '/usr/bin/ccx'
cgx = '/usr/bin/cgx'
gmsh = '/home/cyril/.local/lib/python3.8/site-packages/gmsh-4.8.4-Linux64-sdk/bin/gmsh'
model_name = 'model'


def create_sensor_geometry():
    model = pyc.FeaModel(model_name, ccx, cgx, gmsh)
    model.set_units('m')

    load_lines = []
    constr_lines = []
    part = pyc.Part(model)
    part.goto(0.04, 0)

    [L1, _, _] = part.draw_line_rad(0.01)
    constr_lines.append(L1)
    part.draw_line_ax(0.04)
    part.draw_line_rad(-0.04)
    # part.draw_line_to(20, 40)
    # part.draw_line_to(10, 50)

    part.draw_line_ax(0.01)
    [L1, _, _] = part.draw_line_rad(-0.01)
    load_lines.append(L1)
    [L1, _, _] = part.draw_line_ax(-0.02)
    constr_lines.append(L1)
    part.draw_line_rad(0.04)
    part.draw_line_ax(-0.03)

    return model, part, load_lines, constr_lines


def set_loads_and_constraints(model, load_lines, constr_lines):
    model.set_load('press', load_lines, 63661977)
    model.set_constr('fix', constr_lines[0], 'y')
    model.set_constr('fix', constr_lines[1], 'x')


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
    model.set_etype('plstrain', part, 0.1)
    try:
        model.mesh(elem_size, "esize", 'gmsh')
    except Exception:
        return 'Shit happend'
    return None


def plot_elems_pressures_constraints(model, show_gui):
    model.plot_elements(display=show_gui)
    model.plot_pressures('press')
    model.plot_constraints('constr')


def solve_problem(model):
    mod = pyc.Problem(model, 'struct', 'problem')
    mod.solve()

    disp = True
    fields = 'S2'
    fields = fields.split(',')
    for field in fields:
        fname = model_name + '_' + field
        mod.rfile.nplot(field, fname, display=disp)

    model.view.print_summary()
