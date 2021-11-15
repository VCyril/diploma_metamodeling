import parsing as pars
import plotting as myplt
import matplotlib.pyplot as plt
import finite_elem as fin
import converting_results as conv
import sys

model, part = fin.create_sensor_geometry()
fin.plot_model_geometry(model, False)
fin.set_loads_and_constraints(model, part)
fin.set_mat_props(model, part)

error = fin.set_elem_props(model, part, 'quad', 1)
if error != None:
    print(error)
    sys.exit()

fin.plot_elems_pressures_constraints(model, False)


import time

debu = time.time()

rated_load = [10000, 20000, 30000, 40000, 50000]

f = plt.figure()

ax = f.add_subplot()

rx, er, et = fin.solve_problem(model)
ax.scatter(rx, et, color = 'black')
plt.show()




# *************************************


# list_of_nodes = pars.search_nodes_in_file("model.inp", "NSET=L6", "NSET=L7")
# list_of_coords = pars.get_nodes_coordinates("model.inp", list_of_nodes)
# list_of_strains = pars.get_ex_strain_for_nodes("problem.frd", list_of_nodes)
#
# list_of_nodes, list_of_coords, list_of_strains = conv.convert_results_to_nums(list_of_nodes,
#                                                                               list_of_coords,
#                                                                               list_of_strains)
# sorted_coords_list, sorted_strain_list = conv.sort_by_x_coord(list_of_coords, list_of_strains)
# x_coords, strain_values = conv.make_lists_for_plot(sorted_coords_list, sorted_strain_list)
#
# print(x_coords, list_of_strains)
#
# myplt.plot(x_coords[1:len(x_coords) - 1], strain_values[1:len(x_coords) - 1])
# plt.plot(y_coords, strain_values)
