import itertools


def search_nodes_in_file(file_name, string_from, string_to):
    num_start_line = 0
    num_finish_line = 0
    list_of_results = []

    with open(file_name, 'r') as fp:
        for count, line in enumerate(fp):
            if string_from in line:
                num_start_line = count
            if string_to in line:
                num_finish_line = count
                break
        fp.seek(0, 0)

        for line in itertools.islice(fp, num_start_line + 1, num_finish_line):
            list_of_results.extend(line.rsplit(", "))

    return list(filter(lambda a: a != '\n', list_of_results))


def get_nodes_coordinates(file_name, nodes_nums):
    list_of_coords = []

    with open(file_name, 'r') as fp:
        for line in fp:
            if "*** E L E M E N T S ***" in line:
                break
            line_elem_list = line.rsplit(', ')
            if line_elem_list[0] in nodes_nums:
                list_of_coords.append([line_elem_list[0], line_elem_list[1], line_elem_list[2]])
    return list_of_coords


def get_ex_strain_for_nodes(file_name, node_nums):
    num_start_line = 0
    num_finish_line = 0
    list_of_strains = []
    with open(file_name, 'r') as fp:
        for count, line in enumerate(fp):
            if "TOSTRAIN" in line:
                num_start_line = count
            if "FORC" in line:
                num_finish_line = count
                break
        fp.seek(0, 0)

        for line in itertools.islice(fp, num_start_line + 8, num_finish_line - 3):
            if line[3:13].replace(" ", "") in node_nums:
                list_of_strains.append([line[3:13].replace(" ", ""), line[13:25].replace(" ", "")])

    return list_of_strains

