import parsing as pars
import plotting as myplt
import finite_elem as fin
import converting_results as conv
import sys

model, part, load_lines, const_lines = fin.create_sensor_geometry()
fin.set_loads_and_constraints(model, load_lines, const_lines)
fin.plot_model_geometry(model, False)
fin.set_mat_props(model, part)

error = fin.set_elem_props(model, part, 'quad', 0.001)
if error != None:
    print(error)
    sys.exit()

fin.plot_elems_pressures_constraints(model, False)
fin.solve_problem(model)

list_of_nodes = pars.search_nodes_in_file("model.inp", "NSET=L6", "NSET=L7")
list_of_coords = pars.get_nodes_coordinates("model.inp", list_of_nodes)
list_of_strains = pars.get_ex_strain_for_nodes("problem.frd", list_of_nodes)

list_of_nodes, list_of_coords, list_of_strains = conv.convert_results_to_nums(list_of_nodes,
                                                                              list_of_coords,
                                                                              list_of_strains)
sorted_coords_list, sorted_strain_list = conv.sort_by_x_coord(list_of_coords, list_of_strains)
x_coords, strain_values = conv.make_lists_for_plot(sorted_coords_list, sorted_strain_list)

print(x_coords, list_of_strains)

myplt.plot(x_coords[1:len(x_coords) - 1], strain_values[1:len(x_coords) - 1])
# plt.plot(y_coords, strain_values)
