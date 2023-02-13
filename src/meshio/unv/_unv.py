"""
I/O for IDEAS *.unv files.
"""
import numpy as np
import io

try:
    from ..__about__ import __version__
    from .._common import warn
    from .._exceptions import ReadError
    from .._files import open_file
    from .._helpers import register_format
    from .._mesh import CellBlock, Mesh

except ImportError as e:
    import os
    import sys

    _realpath = os.path.realpath
    _dirname = os.path.dirname

    sys.path.append(_dirname(_dirname(_dirname(_realpath(__file__)))))

    del _realpath
    del _dirname

    from meshio.__about__ import __version__
    from meshio._common import warn
    from meshio._exceptions import ReadError
    from meshio._files import open_file
    from meshio._helpers import register_format
    from meshio._mesh import CellBlock, Mesh


DELIMITER = f"{-1:6n}"
FMT_DTS = "{0:>6n}"
FMT_INT = "{0:10n}"
FMT_FLT = "{0:13.5E}"
FMT_DBL = "{0:25.16E}"
FMT_STR = "{0:<80s}"


unv_to_meshio_dataset = {
    15: "NODE1P",
  2411: "NODE2P",
  2412: "ELEMENT",
    55: "NODAL DATA",
}
meshio_to_unv_dataset = {v: k for k, v in unv_to_meshio_dataset.items()}

unv_to_meshio_type = {
    21: "line",         # Linear Beam                               Edge Lagrange P1
    22: "line3",        # Tapered Beam                              Edge Lagrange P2
    24: "line3",        # Parabolic Beam                            Edge Lagrange P2
    81: "triangle",     # Axisymetric Solid Linear Triangle         Triangle Lagrange P1
    82: "triangle6",    # Axisymetric Solid Parabolic Triangle      Triangle Lagrange P2
    91: "triangle",     # Thin Shell Linear Triangle                Triangle Lagrange P1
    92: "triangle6",    # Thin Shell Parabolic Triangle             Triangle Lagrange P2
    84: "quad",         # Axisymetric Solid Linear Quadrilateral    Quadrilateral Lagrange P1
    85: "quad8",        # Axisymetric Solid Parabolic Quadrilateral Quadrilateral Lagrange P2
    94: "quad",         # Thin Shell Linear Quadrilateral           Quadrilateral Lagrange P1
    95: "quad8",        # Thin Shell Parabolic Quadrilateral        Quadrilateral Lagrange P2
    41: "triangle",     # Plane Stress Linear Triangle              Triangle Lagrange P1
    42: "triangle6",    # Plane Stress Parabolic Triangle           Triangle Lagrange P2
# meshio defaults:
    11: "line",         # Rod                                       Edge Lagrange P1
    44: "quad",         # Plane Stress Linear Quadrilateral         Quadrilateral Lagrange P1
    45: "quad8",        # Plane Stress Parabolic Quadrilateral      Quadrilateral Lagrange P2
   111: "tetra",        # Solid Linear Tetrahedron                  Tetrahedron Lagrange P1
   112: "wedge",        # Solid Linear Wedge                        Wedge Lagrange P1
   115: "hexahedron",   # Solid Linear Brick                        Hexahedron Lagrange P1
   116: "hexahedron20", # Solid Parabolic Brick                     Hexahedron Lagrange P2
   118: "tetra10",      # Solid Parabolic Tetrahedron               Tetrahedron Lagrange P2
}
            # 122: "RBE2"}       # Rigid Element                             Quadrilateral Lagrange P1
meshio_to_unv_type = {v: k for k, v in unv_to_meshio_type.items()}

meshio_to_unv_node_order = {
    "triangle6": [0, 3, 1, 4, 2, 5],
    "tetra10":   [0, 4, 1, 5, 2, 6, 7, 8, 9, 3],
    "quad9":     [0, 4, 1, 7, 8, 5, 3, 6, 2],
    "wedge15":   [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14],
}
unv_to_meshio_node_order = {}
for etype in meshio_to_unv_node_order.keys():
    unv_to_meshio_node_order[etype] = {i: meshio_to_unv_node_order[etype][i] for i in
                                       range(len(meshio_to_unv_node_order[etype]))}
    unv_to_meshio_node_order[etype] = {v: k for k, v in
                                       unv_to_meshio_node_order[etype].items()}
    unv_to_meshio_node_order[etype] = [unv_to_meshio_node_order[etype][i] for i in
                                       sorted(unv_to_meshio_node_order[etype].keys())]

# TODO:
unv_to_meshio_analysis_type = {
     "Unknown":                         "unknown",
     "Static":                          "static",
     "Normal Mode":                     "modal",
     "Complex eigenvalue first order":  "modal complex",
     "Transient":                       "transient",
     "Frequency Response":              "frequency response",
     "Buckling":                        "buckling",
     "Complex eigenvalue second order": "modal comlex",
}
meshio_to_unv_analysis_type = {
    "unknown":            0,
    "static":             1,
    "modal":              2,
    "modal complex":      3,
    "transient":          4,
    "frequency response": 5,
    "buckling":           6,
}

unv_to_meshio_data_characteristic = {
     "Unknown":                 "unknown",
     "Scalar":                  "field",
     "3 DOF Global Vector":     "vector3",
     "6 DOF Global Vector":     "vector6",
     "Symmetric Global Tensor": "tensors",
     "General Global Tensor":   "tensorg",
}
meshio_to_unv_data_characteristic = {
    "unknown":    0,
    "field":      1,
    "vector3":    2,
    "vector6":    3,
    "tensors":    4,
    "tensorg":    5,
}

unv_to_meshio_data_type = {
    "Unknown":                 "unknown",
    "General":                 "general",
    "Stress":                  "stress",
    "Strain":                  "strain",
    "Element Force":           "force",
    "Temperature":             "temperature",
    "Heat Flux":               "heat flux",
    "Strain Energy":           "strain energy",
    "Displacement":            "displacement",
    "Reaction Force":          "reaction",
    "Kinetic Energy":          "kinetic energy",
    "Velocity":                "velocity",
    "Acceleration":            "acceleration",
    "Strain Energy Density":   "strain energy density",
    "Kinetic Energy Density":  "kinetic energy density",
    "Hydro-Static Pressure":   "pressure",
    "Heat Gradient":           "heat",
    "Code Checking Value":     "check",
    "Coefficient Of Pressure": "pressure coefficient",
}
meshio_to_unv_data_type = {
    "unknown":                 0,
    "general":                 1,
    "stress":                  2,
    "strain":                  3,
    "force":                   4,
    "temperature":             5,
    "heat flux":               6,
    "strain energy":           7,
    "displacement":            8,
    "reaction":                9,
    "kinetic energy":         10,
    "velocity":               11,
    "acceleration":           12,
    "strain energy density":  13,
    "kinetic energy density": 14,
    "pressure":               15,
    "heat":                   16,
    "check":                  17,
    "pressure coefficient":   18,
}

unv_to_meshio_value_type = {
    "Real":    "real",
    "Complex": "complex",
}
meshio_to_unv_value_type = {
    "real":    2,
    "complex": 5,
}

unv_point_data_model_type = {
     0: "Unknown",
     1: "Structural",
     2: "Heat Transfer",
     3: "Fluid Flow",
}

unv_point_data_analysis_type = {
     0: "Unknown",
     1: "Static",
     2: "Normal Mode",
     3: "Complex eigenvalue first order",
     4: "Transient",
     5: "Frequency Response",
     6: "Buckling",
     7: "Complex eigenvalue second order",
}

unv_point_data_data_characteristic = {
     0: "Unknown",
     1: "Scalar",
     2: "3 DOF Global Vector",
     3: "6 DOF Global Vector",
     4: "Symmetric Global Tensor",
     5: "General Global Tensor",
}

unv_point_data_value_count = {
    "Scalar": 1,
    "3 DOF Global Vector": 3,
    "6 DOF Global Vector": 6,
    "Symmetric Global Tensor": 6,
    "General Global Tensor": 9,
}

unv_point_data_specific_data_type = {
     0: "Unknown",
     1: "General",
     2: "Stress",
     3: "Strain",
     4: "Element Force",
     5: "Temperature",
     6: "Heat Flux",
     7: "Strain Energy",
     8: "Displacement",
     9: "Reaction Force",
    10: "Kinetic Energy",
    11: "Velocity",
    12: "Acceleration",
    13: "Strain Energy Density",
    14: "Kinetic Energy Density",
    15: "Hydro-Static Pressure",
    16: "Heat Gradient",
    17: "Code Checking Value",
    18: "Coefficient Of Pressure",
}

unv_point_data_data_type = {
     2: "Real",
     5: "Complex",
}


def read(filename):
    """Reads a IDEAS unv file."""
    with open_file(filename, "r") as f:
        out = read_buffer(f)

    return out


def read_buffer(f):
    points = []
    # Initialize the optional data fields
    point_gids = []
    cells = {}
    cell_gids = {}
    nsets = {}
    elsets = {}
    field_data = {}
    cell_data = {}
    point_data = {}
    nsets = {}
    elsets = {}

    while True:
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:          # EOF
            break
        if line != DELIMITER: # comments
            continue
        last_pos = f.tell()   # in dataset, read next line = dataset number
        line = f.readline().strip("\n")
        if not line:          # EOF
            break

        dataset = int(line.strip())              # dataset number
        if dataset in unv_to_meshio_dataset.keys():  # known dataset number
            if unv_to_meshio_dataset[dataset] == "NODE1P":
                _points, _point_gids = _read_sp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            elif unv_to_meshio_dataset[dataset] == "NODE2P":
                _points, _point_gids = _read_dp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            elif unv_to_meshio_dataset[dataset] == "ELEMENT":
                _cells, _cell_gids = _read_cells(f)
                for cell_type in _cells.keys():
                    if cell_type not in cells.keys():
                        cells.setdefault(cell_type, [])
                        cell_gids.setdefault(cell_type, [])
                    cells[cell_type].extend(_cells[cell_type])
                    cell_gids[cell_type].extend(_cell_gids[cell_type])

            # TODO:
            elif unv_to_meshio_dataset[dataset] == "NODAL DATA":
                point_data = _update_dict_of_dicts(point_data, _read_point_data(f))

        # too many datasets to specifically skip them
        else:
            _read_dataset(f)

    # prepare point gids
    point_gids = {point_gids[i]: i for i in range(len(point_gids))}

    # prepare points
    points = np.array(points, dtype=np.float64)

    # renumber cell nodes and cell_gids
    for cell_type in cells.keys():
        cell_count = 0
        for i, cell in enumerate(cells[cell_type]):
            cells[cell_type][i] = [point_gids[gid] for gid in cell]

        cell_gids[cell_type] = {cell_gids[cell_type][i]: i + cell_count
                                for i in range(len(cell_gids[cell_type]))}

        cell_count += len(cells[cell_type])

    cells = [CellBlock(etype, np.array(cells[etype], dtype=np.int32),
                       cell_gids=cell_gids[etype]) for etype in cells.keys()]

    point_data = _process_point_data(point_data, point_gids)

    if read_point_data:
        return Mesh(
            points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data,
            point_sets=nsets, cell_sets=elsets, point_gids=point_gids
        )


def _read_sp_nodes(f):
    points = []
    point_gids = []
    err = ""
    while True:
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before Node end block.\n"
            break
        if line == DELIMITER:
            break
        if len(line) == 79:
            gid = int(line[:10].strip())
            defcsys = int(line[10:20].strip())
            outcsys = int(line[20:30].strip())
            color = int(line[30:40].strip())
            coors = [float(line[40+j*13:40+(j+1)*13]) for j in range(3)]
            points.append(coors)
            point_gids.append(gid)
        else:
            err += f"Wrong record length for Node {gid:n} at position {last_pos+1:n}.\n"

    if err != "":
        raise ReadError(err)

    return points, point_gids


def _read_dp_nodes(f):
    points = []
    point_gids = []
    err = ""
    while True:
        gid = 0
        # first line
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of Node Dataset block.\n"
            break
        if line == DELIMITER:
            break
        if len(line) == 40:
            gid = int(line[:10].strip())
            defcsys = int(line[10:20].strip())
            outcsys = int(line[20:30].strip())
        else:
            err += f"Wrong Record 1 length for Node at position {last_pos + 1:n}.\n"

        # second line
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of Node dataset block.\n"
            break
        if line == DELIMITER:
            if gid == 0:
                err += f"Missing Record 2 for Node at position {last_pos + 1:n}.\n"
            else:
                err += f"Missing Record 2 for Node {gid:n} at position {last_pos + 1:n}.\n"
            break
        if len(line) == 75:
            line = line.replace('D', 'E')
            coors = [float(line[j:j+25]) for j in [0, 25, 50]]
            points.append(coors)
            point_gids.append(gid)
        else:
            if gid == 0:
                err += f"Wrong Record 2 length for Node at position {last_pos + 1:n}.\n"
            else:
                err += f"Wrong Record 2 length for Node {gid:n} at position {last_pos + 1:n}.\n"

    if err != "":
        raise ReadError(err)

    return points, point_gids


def _read_cells(f):
    cells = {}
    cell_gids = {}
    err = ""
    while True:
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of Element block.\n"
            break
        if line == DELIMITER:
            break

        # element header line
        if len(line) == 60:
            cid = int(line[:10].strip())
            FEid = int(line[10:20].strip())
            etype = unv_to_meshio_type[FEid]
            pid = int(line[20:30].strip())
            mid = int(line[30:40].strip())
            color = int(line[40:50].strip())
            numnodes = int(line[50:60].strip())
            pid = []
        else:
            err += f"Wrong length of Record 1 for element at position {last_pos + 1:n}.\n"

        # TODO:
        # BEAM elements
        if FEid in [21, 22, 24]:
            last_pos = f.tell()
            line = f.readline().strip("\n")
            if not line:
                err += f"File ended before the end of Element block.\n"
                break
            if line == DELIMITER:
                err += f"Missing Record 2 for element {cid:n} at position {last_pos + 1:n}.\n"
                break
            beamdef = [int(line[i*10:(i+1)*10].strip()) for i in range(3)]

        idx = []
        while len(idx) < numnodes:
            last_pos = f.tell()
            line = f.readline().strip("\n")
            if not line:
                err += f"File ended before the end of Element block.\n"
                break
            if line == DELIMITER:
                if FEid in [21, 22, 24]:
                    err += f"Missing Record 3 for element {cid:n} at position {last_pos + 1:n}.\n"
                else:
                    err += f"Missing Record 2 for element {cid:n} at position {last_pos + 1:n}.\n"
                break
            for i in range(0, len(line)-1, 10):
                idx.append(int(line[i:i+10]))

        if FEid not in unv_to_meshio_type.keys():
            err += f"Wrong type of element {cid:n} at position {last_pos:n}.\n"
            continue

        cell_type = unv_to_meshio_type[FEid]
        if cell_type not in cells.keys():
            cells.setdefault(cell_type, [])
            cell_gids.setdefault(cell_type, [])

        # TODO:
        # check the logic
        if cell_type in unv_to_meshio_node_order.keys():
            idx = [idx[i] for i in unv_to_meshio_node_order[etype]]

        cells[cell_type].append(idx)
        cell_gids[cell_type].append(cid)

    if err != "":
        raise ReadError(err)

    return cells, cell_gids


def _update_dict_of_dicts(base: dict, data: dict) -> dict:
    """
    Merges data from one multilevel dictionary to another one.
    At the lowest level the data are overwritten.

    Example:
    -------
    >>> a = {'a': {'b': {'c': [1, 2, 3]}}}
    >>> b = {'a': {'b': {'d': [4, 5, 6]}}}
    >>> c = {'a': {'b': {'d': [7, 8, 9]}}}

    >>> print(f"a =     {a}")
    a =     {'a': {'b': {'c': [1, 2, 3]}}}

    >>> a = _update_dict_of_dicts(a, b)
    >>> print(f"a + b = {a}")
    a + b = {'a': {'b': {'c': [1, 2, 3], 'd': [4, 5, 6]}}}

    >>> a = _update_dict_of_dicts(a, c)
    >>> print(f"a + c = {a}")
    a + c = {'a': {'b': {'c': [1, 2, 3], 'd': [7, 8, 9]}}}
    """
    for key in data.keys():
        if type(data[key]) is not dict or key not in base.keys():
            base.update(data)
        else:
            base[key] = _update_dict_of_dicts(base[key], data[key])

    return base


def read_point_data(f):
    points = []
    # Initialize the optional data fields
    point_gids = []
    point_data = {}

    while True:
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:          # EOF
            break
        if line != DELIMITER: # comments
            continue
        last_pos = f.tell()   # in dataset, read next line = dataset number
        line = f.readline().strip("\n")
        if not line:          # EOF
            break

        dataset = int(line.strip())              # dataset number
        if dataset in unv_to_meshio_dataset.keys():  # known dataset number
            # read points to get their IDs
            if unv_to_meshio_dataset[dataset] == "NODE1P":
                _points, _point_gids = _read_sp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            # read points to get their IDs
            elif unv_to_meshio_dataset[dataset] == "NODE2P":
                _points, _point_gids = _read_dp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            # TODO:
            elif unv_to_meshio_dataset[dataset] == "NODAL DATA":
                point_data = _update_dict_of_dicts(point_data, _read_point_data(f))

        # too many datasets to specifically skip them
        else:
            _read_dataset(f)

    # prepare point gids
    point_gids = {point_gids[i]: i for i in range(len(point_gids.keys()))}

    point_data = _process_point_data(point_data, point_gids)

    return point_data


def _process_point_data(point_data: dict, point_gids: dict):
    # TODO:
    # renumber point_data according to point_gids
    for lcase in point_data.keys():
        for lstep in point_data[lcase].keys():
            data_gids = point_data[lcase][lstep]["gids"]
            data_gids = {data_gids[i]: i for i in range(len(data_gids))}
            # insert zero values to points that were not written in dataset 55
            # and order point data the same as points
            values = point_data[lcase][lstep]["values"]
            numvals = values.shape[1]
            data = []
            for gid in point_gids.keys():
                if gid in data_gids.keys():
                    data.append(values[data_gids[gid]])
                else:
                    data.append([0.] * numvals)
            data = np.array(data, dtype=np.float64)
            point_data[lcase][lstep]["values"] = data
            point_data[lcase][lstep]["gids"] = np.array(list(point_gids.keys()), dtype=np.int32)

    return point_data


def _read_point_data(f) -> dict:
    point_data = {}

    point_data_header = _read_point_data_header(f)
    point_data_values = _read_point_data_lines(f, point_data_header["load case"],
                                                  point_data_header["value count"],
                                                  point_data_header["data type"])

    lcase = point_data_header["load case"]
    step = "{0:10n} - {1:13.5E}".format(point_data_header["load step"],
                                        point_data_header["step value"])
    point_data[lcase] = {step: {"header": point_data_header,
                                "values": np.array(list(point_data_values.values()), dtype=float),
                                "gids":   list(point_data_values.keys())}}
    return point_data


def _read_line(f, err: str) -> (str, int, str):
    last_pos = f.tell()
    line = f.readline().strip("\n")
    _err = ""
    if not line:
        _err = "File ended before and and of Dataset block."
    elif line == DELIMITER:
        _err = err
    return line, last_pos, _err


def _read_point_data_header(f):
    """
    Function to read Nodal Data dataset header
    """
    point_data_header = {}

    err = ""
    for h in range(1): # process first 8 records = result dataset header
        description = []
        # read description - records 1 to 5
        for i in range(5):
            line, last_pos, _err = _read_line(f, "Dataset 55 missing records 1-5.")
            if _err != "":
                err += _err + "\n"
                break
            description.append(line.strip())
        if err != "":
            break
        description = " ".join([d for d in description if d != "" or d != "NONE"])

        # read data defintion - record 6
        line, last_pos, _err = _read_line(f, "Dataset 55 record 6 missing.")
        if _err != "":
            err += _err + "\n"
            break
        dd = [line[j*10:(j+1)*10] for j in range(6)]
        dd = [int(line[j*10:(j+1)*10].strip()) for j in range(6)]
        model_type = unv_point_data_model_type[dd[0]]
        analysis_type = unv_point_data_analysis_type[dd[1]]
        data_characteristic = unv_point_data_data_characteristic[dd[2]]
        specific_data_type = unv_point_data_specific_data_type[dd[3]]
        data_type = unv_point_data_data_type[dd[4]]
        numvals = dd[5]

        # read data defintion - record 7
        # TODO:
        # what if numints > 6 ?
        line, last_pos, _err = _read_line(f, "Dataset 55 missing record 7.")
        if _err != "":
            err += _err + "\n"
            break
        numints = int(line[:10].strip())
        numreals = int(line[10:20].strip())
        ints = []
        line = line[20:]
        while len(ints) < numints:
            ints += [int(line[(i*10):((i+1)*10)].strip()) for i in range(int(len(line) / 10))]
            if len(ints) < numints:
                line, last_pos, _err = _read_line(f, "Dataset 55 incomplete record 7.")
                if _err != "":
                    err += _err + "\n"
                    break
        if _err != "":
            err += _err + "\n"
            break

        # read data defintion - record 8
        # TODO:
        # what if numreals > 6 ?
        reals = []
        while len(reals) < numreals:
            line, last_pos, _err = _read_line(f, "Dataset 55 missing record 8.")
            if _err != "":
                err += _err + "\n"
                break
            reals += [float(line[i*13:(i+1)*13].strip()) for i in range(int(len(line) / 13))]

        # process it
        if numints == 1:
            point_data_lcase = ints[0] # lcase ID
        elif numints > 1:
            point_data_lcase = ints[0] # lcase ID
            point_data_step = ints[1]
        else:
            point_data_lcase = 0
            point_data_step = 0

        # TODO:
        # not sure about the second order, might be general analysis type records
        if analysis_type in ("Complex eigenvalue first order",
                             "Complex eigenvalue second order"):
            point_data_stepval = complex(reals[0], reals[1])
        elif numreals > 0:
            point_data_stepval = reals[0]
        else:
            point_data_stepval = 0.0

        break

    if err != "":
        raise ReadError(err)

    point_data_header = {
        # "model type": model_type,
        "analysis type": unv_to_meshio_analysis_type[analysis_type],
        "data characteristic": unv_to_meshio_data_characteristic[data_characteristic],
        "specific data type": unv_to_meshio_data_type[specific_data_type],
        "data type": unv_to_meshio_value_type[data_type],
        "value count": numvals,
        "load case": point_data_lcase,
        "load step": point_data_step,
        "step value": point_data_stepval,
    }

    return point_data_header


def _read_point_data_lines(f, lcase: int, numvals: int=3, valtype: str="Real") -> dict:
    """
    Function to read Nodal Data dataset header
    """
    if valtype == "Complex":
        numvals *= 2

    point_data = {}

    err = ""
    # read point data
    while True:
        # read node ID
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of Dataset 55 (Load Case {lcase:n}) block.\n"
            break
        if line == DELIMITER:
            break

        point_id = int(line[:10].strip())

        data = []
        while len(data) < numvals:
            line, last_pos, _err = _read_line(f, f"Dataset 55 (Load Case {lcase:n}) " +
                                                 f"Node {point_id:n} data of wrong length.")
            if _err != "":
                err += _err + "\n"
                break

            data += [float(line[i*13:(i+1)*13].strip()) for i in range(int(len(line) / 13))]

        if valtype == "Complex":
            data = [complex(data[i], data[i+1]) for i in range(0, numvals, 2)]

        point_data.setdefault(point_id, data)

    return point_data


def _read_dataset(f):
    """
    Dummy function to skip unknown datasets
    """
    err = ""
    while True:
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of dataset block.\n"
            break
        if line == DELIMITER:
            break

    if err != "":
        raise ReadError(err)

    return


def _write_sp_nodes(points: np.ndarray, node_gids: dict=None) -> str:
    defsys = 1
    outsys = 1
    color = 1

    dataset = DELIMITER + "\n"
    dataset += FMT_DTS.format(meshio_to_unv_dataset["NODE1P"]) + "\n"

    for i, coor in enumerate(points):
        if node_gids is not None:
            dataset += f"{node_gids[i]:10n}{defsys:10n}{outsys:10n}{color:10n}"
        else:
            dataset += f"{i+1:10n}{defsys:10n}{outsys:10n}{color:10n}"
        dataset += f"{coor[0]:13.5E}{coor[1]:13.5E}{coor[2]:13.5E}\n"

    dataset = DELIMITER + "\n"

    return dataset


def _write_dp_nodes(points: np.ndarray, node_gids: dict=None) -> str:
    defsys = 1
    outsys = 1
    color = 1

    dataset = DELIMITER + "\n"
    dataset += FMT_DTS.format(meshio_to_unv_dataset["NODE2P"]) + "\n"

    for i, coor in enumerate(points):
        if node_gids is not None:
            dataset += f"{node_gids[i]:10n}{defsys:10n}{outsys:10n}{color:10n}\n"
        else:
            dataset += f"{i+1:10n}{defsys:10n}{outsys:10n}{color:10n}\n"
        dataset += f"{coor[0]:25.16E}{coor[1]:25.16E}{coor[2]:25.16E}\n".replace("E", "D")

    dataset += DELIMITER + "\n"

    return dataset


def _write_line(values: [list | np.ndarray], maxval: int, fmt: str) -> str:
    line = ""
    for i in range(len(values)):
        line += fmt.format(values[i])
        if (i + 1) % maxval == 0:
            line += "\n"
    if not line.endswith("\n"):
        line += "\n"

    return line


def _write_elements(cells: list, node_gids: dict = None) -> str:
    pid = 1
    mid = 1
    color = 1

    dataset = DELIMITER + "\n"
    dataset += FMT_DTS.format(meshio_to_unv_dataset['ELEMENT']) + "\n"

    cid = 0
    for i, cell_block in enumerate(cells):
        cell_gids = cell_block.cell_gids
        etype = cell_block.type
        FEid = meshio_to_unv_type[etype]

        if cell_gids is not None:
            element_gids = {v: k for k, v in cell_gids.items()}

        for j, points in enumerate(cell_block.data):
            if cell_gids is not None:
                eid = element_gids[cid]
            else:
                eid = cid + 1
            numnodes = len(points)
            dataset += f"{eid:10n}{FEid:10n}{pid:10n}{mid:10n}{color:10n}{numnodes:10n}\n"

            if node_gids is not None:
                nodes = [node_gids[p] for p in points]
            else:
                nodes = [point + 1 for point in points]

            if etype in unv_to_meshio_node_order.keys():
                nodes = [nodes[i] for i in meshio_to_unv_node_order[etype]]

            # beam elements
            if FEid in (21, 22, 24):
                #         orientation node,  endA ID, endB ID
                dataset += f"{0:10n}{0:10n}{0:10n}\n"

            # nodes
            dataset += _write_line(nodes, 8, FMT_INT)
            # for i in range(numnodes):
            #     dataset += f"{nodes[i]:10n}"
            #     if (i + 1) % 8 == 0:
            #         dataset += "\n"
            # if not dataset.endswith("\n"):
            #     dataset += "\n"
            cid += 1

    dataset += DELIMITER + "\n"

    return dataset

# TODO:
def _write_nodal_data(point_data: np.ndarray, header: dict, point_gids: dict=None) -> str:
    dataset = DELIMITER + "\n"
    dataset += FMT_DTS.format(meshio_to_unv_dataset["NODAL DATA"]) + "\n"
    for i in range(5):
        dataset += "NONE\n"

    mt = 1
    at = meshio_to_unv_analysis_type[header["analysis type"]]
    dc = meshio_to_unv_data_characteristic[header["data characteristic"]]
    sd = meshio_to_unv_data_type[header["specific data type"]]
    dt = meshio_to_unv_value_type[header["data type"]]
    vc = header["value count"]

    lc = header["load case"]
    ls = header["load step"]
    sv = header["step value"]

    for val in (mt, at, dc, sd, dt, vc):
        dataset += FMT_INT.format(val)
    dataset += "\n"

    if at in (0, 1, 6): # unknown, static, buckling
        for val in (1, 1, lc):
            dataset += FMT_INT.format(val)
        dataset += "\n"
        dataset += FMT_FLT.format(sv) + "\n"
    elif at == 2:       # modal
        for val in (2, 4, lc, ls):
            dataset += FMT_INT.format(val)
        dataset += "\n"
        for val in (sv, 0.0, 0.0, 0.0):
            dataset += FMT_FLT.format(val)
        dataset += "\n"
    elif at == 3:       # complex modal
        for val in (2, 6, lc, ls):
            dataset += FMT_INT.format(val)
        dataset += "\n"
        for val in (sv.real, sv.imag, 0.0, 0.0, 0.0, 0.0):
            dataset += FMT_FLT.format(val)
        dataset += "\n"
    elif at in (4, 5):  # transient, frequency response
        for val in (2, 1, lc, ls):
            dataset += FMT_INT.format(val)
        dataset += "\n"
        dataset += FMT_FLT.format(sv) + "\n"
    else:               # general format
        for val in (2, 1, lc, ls):
            dataset += FMT_INT.format(val)
        dataset += "\n"
        dataset += FMT_FLT.format(sv) + "\n"

    if point_gids is None:
        node_gids = {i: i + 1 for i in range(len(point_data))}
    else:
        node_gids = {v: k for k, v in point_gids.items()}

    for i, row in enumerate(point_data):
        dataset += FMT_INT.format(node_gids[i]) + "\n"
        for j, val in enumerate(row):
            dataset += FMT_FLT.format(val)
            if (j + 1) % 6 == 0:
                dataset += "\n"
        if not dataset.endswith("\n"):
            dataset += "\n"

    dataset += DELIMITER + "\n"

    return dataset


def _write_point_data(point_data: np.ndarray, point_gids: list=None) -> str:
    datasets = ""
    # TODO:
    if "header" not in point_data.keys():
        analysis_type = "unknown"
        # header = {"analysis type": "unknown",
        #           "data chara

    for lcase in point_data.keys():
        for load_step in point_data[lcase].keys():
            datasets += _write_nodal_data(point_data[lcase][load_step], point_gids)

    return datasets


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        warn(
            "IDEAS UNV requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    with open_file(filename, "wt") as f:
        point_gids = mesh.point_gids
        node_gids = None if mesh.point_gids is None else {v: k for k, v in mesh.point_gids.items()}
        f.write("IDEAS unv file format\n")
        f.write(f"written by meshio v{__version__}\n")

        if points is not None and len(points) > 0:
            f.write(_write_dp_nodes(points, node_gids))
        if mesh.cells is not None and len(mesh.cells) > 0:
            f.write(_write_elements(mesh.cells, node_gids))

        # TODO:
        # point sets
        # cell sets
        # point data


register_format(
    "unv", [".unv", ".unv.gz"], read, {"unv": write}
)

if __name__ == "__main__":
    point_data = np.random.rand(100, 3)
    header = {
        "analysis type":             "static",
        "data characteristic":      "vector3",
        "specific data type":  "displacement",
        "data type":                   "real",
        "value count":                      3,
        "load case":                      101,
        "load step":                        1,
        "step value":                     0.0,
    }

    dataset = _write_nodal_data(point_data, header)
    print(dataset)


    # mesh = read("./res/hex_double_in.unv")
    # print(mesh)
    # # print(point_data)
    # write("./res/hex_double_out.unv", mesh)
    # from meshio import vtk
    # id = 0
    # for lcase, lcase_data in mesh.point_data.items():
    #     for lstep, lstep_data in lcase_data.items():
    #         # reformated = dict()
    #         # reformated["x"] = lstep_data["values"][:,0]
    #         # reformated["Y"] = lstep_data["values"][:,1]
    #         # reformated["Z"] = lstep_data["values"][:,2]

    #         # print(f"{reformated = }")

    #         mesh.point_data = {f"{lstep_data['header']['step value']:.5E}": lstep_data["values"]}

    #         vtk.write(f"./res/hex_double_out_{id+1:04n}.vtk", mesh, "4.2", binary=False)
    #         id += 1

