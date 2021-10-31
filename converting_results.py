from operator import itemgetter


def convert_results_to_nums(list_of_nodes, list_of_coords, list_of_strains):
    list_of_nodes = list(map(int, list_of_nodes))

    for i, sublist in enumerate(list_of_coords):
        list_of_coords[i][0] = int(sublist[0])
        list_of_coords[i][1] = float(sublist[1])
        list_of_coords[i][2] = float(sublist[2])
    for i, sublist in enumerate(list_of_strains):
        list_of_strains[i][0] = int(sublist[0])
        list_of_strains[i][1] = float(sublist[1])

    return list_of_nodes, list_of_coords, list_of_strains


def sort_by_x_coord(list_of_coords, list_of_strains):
    sorted_list_of_coords = sorted(list_of_coords, key=itemgetter(1))

    sorted_list_of_strains = []
    for coord in sorted_list_of_coords:
        index = list_of_coords.index(coord)
        sorted_list_of_strains.append(list_of_strains[index])

    return sorted_list_of_coords, sorted_list_of_strains


def make_lists_for_plot(sorted_list_of_coords, sorted_list_of_strains):
    x_coords = []
    strain_values = []
    for value in sorted_list_of_coords:
        x_coords.append(value[1])
    for value in sorted_list_of_strains:
        strain_values.append(value[1])

    return x_coords, strain_values
