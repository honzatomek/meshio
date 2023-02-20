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
    from meshio._mesh import CellBlock, Mesh, LoadCase, LoadStep, Data


DELIMITER = f"{-1:6n}"
_FMT_DTS = "{0:>6n}"               # format dataset number
_FMT_INT = "{0:10n}"               # single precision integer
_FMT_SPR = "{0:13.5E}"             # single precision real value
_FMT_DPR = "{0:25.16E}"            # double precision real value
_FMT_SPC = "{0:13.5E}{1:13.5E}"    # single precision complex value
_FMT_DPC = "{0:25.16E}{1:25.16E}"  # single precision complex value
_FMT_STR = "{0:<80s}"              # string of 80 chars

FMT_DTS = lambda x: _FMT_DTS.format(x)                # format dataset number
FMT_INT = lambda x: _FMT_INT.format(x)                # single precision integer
FMT_SPR = lambda x: _FMT_SPR.format(x)                # single precision real value
FMT_DPR = lambda x: _FMT_DPR.format(x)                # double precision real value
FMT_SPC = lambda x: _FMT_SPC.format(x.real, x.imag)   # single precision real value
FMT_DPC = lambda x: _FMT_DPC.format(x.real, x.imag)   # double precision real value
FMT_STR = lambda x: _FMT_STR.format(x).rstrip()       # string


unv_to_meshio_dataset = {
    15: "point",
  2411: "point", # default
  2412: "cell",
    55: "point data",
}
meshio_to_unv_dataset = {v: k for k, v in unv_to_meshio_dataset.items()}

unv_to_meshio_element_type = {
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
meshio_to_unv_element_type = {v: k for k, v in unv_to_meshio_element_type.items()}

meshio_to_unv_node_order = {
    "triangle6": [0, 3, 1, 4, 2, 5],
    "tetra10":   [0, 4, 1, 5, 2, 6, 7, 8, 9, 3],
    "quad9":     [0, 4, 1, 7, 8, 5, 3, 6, 2],
    "wedge15":   [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14],
}
unv_to_meshio_node_order = {}
for etype in meshio_to_unv_node_order.keys():
#     unv_to_meshio_node_order[etype] = {i: meshio_to_unv_node_order[etype][i] for i in
#                                        range(len(meshio_to_unv_node_order[etype]))}
#     unv_to_meshio_node_order[etype] = {v: k for k, v in
#                                        unv_to_meshio_node_order[etype].items()}
    unv_to_meshio_node_order[etype] = {meshio_to_unv_node_order[etype][i]: i for i in
                                       range(len(meshio_to_unv_node_order[etype]))}
    unv_to_meshio_node_order[etype] = [unv_to_meshio_node_order[etype][i] for i in
                                       sorted(unv_to_meshio_node_order[etype].keys())]

unv_to_meshio_analysis = {
    0: "unknown",
    1: "static",
    2: "modal",
    3: "modal complex",
    4: "transient",
    5: "frequency response",
    6: "buckling",
    7: "modal complex",
    9: "nlstatic",
}
meshio_to_unv_analysis = {v: k for k, v in unv_to_meshio_analysis.items()}

unv_to_meshio_data_character = {
    0: "unknown",
    1: "scalar",
    2: "vector3",
    3: "vector6",
    4: "symmetric tensor",
    5: "general tensor",
    6: "stress resultant",
}
meshio_to_unv_data_character = {v: k for k, v in unv_to_meshio_data_character.items()}

unv_to_meshio_data_type = {
     0: "unknown",
     1: "general",
     2: "stress",
     3: "strain",
     4: "force",
     5: "temperature",
     6: "heat flux",
     7: "strain energy",
     8: "displacement",
     9: "reaction",
    10: "kinetic energy",
    11: "velocity",
    12: "acceleration",
    13: "strain energy density",
    14: "kinetic energy density",
    15: "pressure",
    16: "heat",
    17: "check",
    18: "pressure coefficient",
    19: "ply stress",
    20: "ply strain",
    21: "cell scalar",
    22: "cell scalar",
    23: "reaction heat flow",
    24: "stress error density",
    25: "stress variation",
    27: "shell and plate elem stress resultant",
    28: "length",
    29: "area",
    30: "volume",
    31: "mass",
    32: "constraint forces",
    34: "plastic strain",
    35: "creep strain",
    36: "strain energy error",
    37: "dynamic stress at nodes",
    93: "cell unknown",
    94: "cell scalar",
    95: "cell vector3",
    96: "cell vector6",
    97: "cell symmetric tensor",
    98: "cell global tensor",
    99: "cell shell and plate resultant",
}
meshio_to_unv_data_type = {v: k for k, v in unv_to_meshio_data_type.items()}

unv_to_meshio_value_type = {
     1: "integer",
     2: "sp real",
     4: "dp real",
     5: "sp complex",
     6: "dp complex",
}
meshio_to_unv_value_type = {v: k for k, v in unv_to_meshio_value_type.items()}
meshio_to_unv_value_type["real"] =    2
meshio_to_unv_value_type["complex"] = 5
unv_to_meshio_value_dtype = {
     1: int,
     2: float,
     4: np.float128,
     5: np.complex128,
     6: np.complex256,
}
meshio_to_unv_value_dtype = {
     "integer":     int,
     "real":        int,
     "sp real":     float,
     "dp real":     np.float128,
     "complex":     float,
     "sp complex":  np.complex128,
     "dp complex":  np.complex256,
}
meshio_to_unv_value_format = {
     "integer":     FMT_INT,
     "real":        FMT_SPR,
     "sp real":     FMT_SPR,
     "dp real":     FMT_DPR,
     "complex":     FMT_SPC,
     "sp complex":  FMT_SPC,
     "dp complex":  FMT_DPC,
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
            if dataset == 15:                    # single precision node
                _points, _point_gids = _read_sp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            elif dataset == 2411:                # double precision node
                _points, _point_gids = _read_dp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            elif dataset == 2412:                # elements
                _cells, _cell_gids = _read_cells(f)
                for cell_type in _cells.keys():
                    if cell_type not in cells.keys():
                        cells.setdefault(cell_type, [])
                        cell_gids.setdefault(cell_type, [])
                    cells[cell_type].extend(_cells[cell_type])
                    cell_gids[cell_type].extend(_cell_gids[cell_type])

            # TODO:
            elif dataset == 55:                 # nodal data
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

    # TODO:
    point_data = _process_point_data(point_data, point_gids)

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
            etype = unv_to_meshio_element_type[FEid]
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

        if FEid not in unv_to_meshio_element_type.keys():
            err += f"Wrong type of element {cid:n} at position {last_pos:n}.\n"
            continue

        cell_type = unv_to_meshio_element_type[FEid]
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


def _process_point_data(point_data: dict, point_gids: dict):
    # TODO:
    # renumber point_data according to point_gids
    for lc in point_data.keys():
        loadsteps = point_data[lc]
        for ls in [k for k in loadsteps.keys() if k != "analysis"]:
            loadstep = loadsteps[ls]
            for dt in [k for k in loadstep.keys() if k != "value"]:
                data_type = loadstep[dt]
                gids = [point_gids[gid] for gid in data_type["gids"]]
                point_data[lc][ls][dt]["gids"] = np.array(gids, dtype=int)
                # TODO:
                # point data for a point, that is not specified is implied to
                # be zero
    return point_data


def _read_point_data(f) -> dict:
    data = {}

    header = _read_point_data_header(f)
    print(f"{header = }")
    gids, values = _read_point_data_lines(f, header["load case"],
                                             header["value count"],
                                             header["value type"])

    load_case = header["load case"]
    load_step = header["load step"]
    step_val  = header["step value"]
    data_type = header["data type"]
    data_char = header["data character"]

    # TODO:
    point_data = {"character": data_char,               # scalar, vector3, tensor, ..
                  "value type": header["value type"],   # not necessary, can be inferred
                  "gids": gids,                         # original node IDs -> renumbering
                  "data": values}                       # values

    lstep = {"value": step_val,
             data_type: point_data}

    lcase = {load_step: lstep}

    analysis = {"analysis": header["analysis"],
                load_case: lcase}
    #
    # !!! Only for POINT DATA !!!
    #
    # -> analysis[load case][load step][data type][data] = [..., ..., ...]
    # -> analysis: analysis: analysis type - str
    #              load case 1: load case ID - (int, dict)
    #              load case 2: load case ID - (int, dict)
    #
    # -> load case 1: load step 1 - (int, dict)
    #                 load step 2 - (int, dict)
    #
    # -> load step 1: value       - float (e.g. time, eigenfrequency etc.)
    #                 data type 1 - (str, dict) - (e.g. stress, strain, force)
    #                 data type 2 - (str, dict)
    #
    # -> data type 1: data character - str - (e.g scalar, vector3, tensor)
    #                 value type     - str - (e.g. integer, sp real, dp complex, ...)
    #                 gids           - list - original Node IDs
    #                 data           - np.ndarray(value type) - values
    #

    return analysis


def _read_line(f, err: str) -> (str, int, str):
    last_pos = f.tell()
    line = f.readline().strip("\n")
    _err = ""
    if not line:
        _err = "File ended before and and of Dataset block.\n"
    elif line == DELIMITER:
        _err = err
    return line, last_pos, _err


def _read_point_data_header(f):
    """
    Function to read Nodal Data dataset header
    """
    header = {}

    err = ""
    for h in range(1): # Process first 8 records = result dataset header,
                       # cycle is used for the use of break command.
                       # cylce once
        description = []
        # read description - records 1 to 5
        for i in range(5):
            line, last_pos, err = _read_line(f, "Dataset 55 missing records 1-5.")
            if err != "": break      # break inner loop in case of read error

            description.append(line[:80].strip())
        if err != "": break          # break outer loop in case of read error

        description = "\n".join([d for d in description if d != "" and d.upper() != "NONE"])

        # read data defintion - record 6
        line, last_pos, err = _read_line(f, "Dataset 55 record 6 missing.")
        if err != "": break          # break outer loop in case of read error

        # TODO:
        dd = [line[j*10:(j+1)*10] for j in range(6)]
        dd = [int(line[j*10:(j+1)*10].strip()) for j in range(6)]
        # model_type = unv_point_data_model_type[dd[0]]
        analysis = unv_to_meshio_analysis[dd[1]]
        data_character = unv_to_meshio_data_character[dd[2]]
        data_type = unv_to_meshio_data_type[dd[3]]
        value_type = unv_to_meshio_value_type[dd[4]]
        numvals = dd[5]

        # read data defintion - record 7
        line, last_pos, err = _read_line(f, "Dataset 55 missing record 7.")
        if err != "": break          # break outer loop in case of read error

        numints = int(line[:10].strip())
        numreals = int(line[10:20].strip())
        ints = []
        line = line[20:]
        while len(ints) < numints:
            ints += [int(line[(i*10):((i+1)*10)].strip()) for i in range(int(len(line) / 10))]
            if len(ints) < numints:  # continue reading lines until numints is satisfied
                line, last_pos, err = _read_line(f, "Dataset 55 (Load Case {ints[0]:n}) "
                                                    "incomplete record 7.")
                if err != "": break  # break inner loop in case of read error
        if err != "": break          # break outer loop in case of read error

        # read data defintion - record 8
        reals = []
        while len(reals) < numreals:
            line, last_pos, err = _read_line(f, "Dataset 55 missing record 8.")
            if err != "": break      # break inner loop in case of read error
            reals += [float(line[i*13:(i+1)*13].strip()) for i in range(int(len(line) / 13))]
        if err != "": break          # break outer loop in case of read error

        # process it
        if numints == 1:
            lcase = ints[0] # lcase ID
            step = 0
        elif numints > 1:
            lcase = ints[0] # lcase ID
            step = ints[1]
        else:
            lcase = 0
            step = 0

        # TODO:
        # not sure about the second order, might be general analysis type records
        if analysis == "modal complex":
            stepval = complex(reals[0], reals[1])
        elif numreals > 0:
            stepval = reals[0]
        else:
            stepval = 0.0

        break                        # not neccessary

    if err != "":
        raise ReadError(err)

    # header = {
    #     # "model type": model_type,
    #     "analysis": unv_to_meshio_analysis[analysis],
    #     "data character": unv_to_meshio_data_character[data_character],
    #     "data type": unv_to_meshio_data_type[data_type],
    #     "value type": unv_to_meshio_value_type[value_type],
    #     "value count": numvals,
    #     "load case": lcase,
    #     "load step": step,
    #     "step value": stepval,
    # }
    header = {
        "analysis": analysis,
        "data character": data_character,
        "data type": data_type,
        "value type": value_type,
        "value count": numvals,
        "load case": lcase,
        "load step": step,
        "step value": stepval,
    }

    return header


def _read_point_data_lines(f, lcase: int, numvals: int=3, value_type: str="real") -> dict:
    """
    Function to read Nodal Data dataset header
    In:
    """
    vallen = len(meshio_to_unv_value_format[value_type](1))
    if "complex" in value_type:
        numvals *= 2
        vallen /= 2
    dtype = meshio_to_unv_value_dtype[value_type]

    point_data = []
    point_gids = []                    # original node IDs, needed some nodes can
                                       # be skipped

    err = ""
    while True:                        # read point data each loop one node
        # read node ID
        last_pos = f.tell()
        line = f.readline().strip("\n")
        if not line:
            err += f"File ended before the end of Dataset 55 (Load Case {lcase:n}) block."
            break
        if line == DELIMITER:
            break

        pointid = int(line[:10].strip())

        data = []
        while len(data) < numvals:
            line, last_pos, err = _read_line(f, f"Dataset 55 (Load Case {lcase:n}) " +
                                                f"Node {pointid:n} data of wrong length.")
            if err != "": break    # break inner loop in case of error

            data += [float(line[i*vallen:(i+1)*vallen].strip()) for i in range(int(len(line) / vallen))]

        if err != "": break        # break outer loop in case of error

        if "complex" in value_type:
            data = [complex(data[i], data[i+1]) for i in range(0, numvals, 2)]

        point_data.append(data)
        point_gids.append(pointid)

    if err != "":
        raise ReadError(err)

    # return original node IDs and a numpy array
    return point_gids, np.array(point_data, dtype=dtype)


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
    dataset += FMT_DTS(15) + "\n"

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
    dataset += FMT_DTS(meshio_to_unv_dataset["point"]) + "\n"

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
        if type(fmt) is str:
            line += fmt.format(values[i])
        else:
            line += fmt(values[i])
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
    dataset += FMT_DTS(meshio_to_unv_dataset["cell"]) + "\n"

    cid = 0
    for i, cell_block in enumerate(cells):
        cell_gids = cell_block.cell_gids
        etype = cell_block.type
        FEid = meshio_to_unv_element_type[etype]

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
    dataset += FMT_DTS(meshio_to_unv_dataset["NODAL DATA"]) + "\n"
    for i in range(5):
        dataset += "NONE\n"

    mt = 1
    # at = meshio_to_unv_analysis_type[header["analysis type"]]
    # dc = meshio_to_unv_data_characteristic[header["data characteristic"]]
    # sd = meshio_to_unv_data_type[header["specific data type"]]
    # dt = meshio_to_unv_value_type[header["data type"]]
    # vc = header["value count"]

    at = meshio_to_unv_analysis[header["analysis"]]
    dc = meshio_to_unv_data_character[header["character"]]
    sd = meshio_to_unv_data_type[header["data type"]]
    dt = meshio_to_unv_value_type[header["value type"]] # real = 2, complex = 5
    vc = point_data.shape[1] * (2 if dt == 5 else 1)

    lc = header["load case"]
    ls = header["load step"]
    sv = header["step value"]

    for val in (mt, at, dc, sd, dt, vc):
        dataset += FMT_INT(val)
    dataset += "\n"

    if at in (0, 1, 6): # unknown, static, buckling
        for val in (1, 1, lc):
            dataset += FMT_INT(val)
        dataset += "\n"
        dataset += FMT_SPR(sv) + "\n"
    elif at == 2:       # modal
        for val in (2, 4, lc, ls):
            dataset += FMT_INT(val)
        dataset += "\n"
        for val in (sv, 0.0, 0.0, 0.0):
            dataset += FMT_SPR(val)
        dataset += "\n"
    elif at == 3:       # complex modal
        for val in (2, 6, lc, ls):
            dataset += FMT_INT(val)
        dataset += "\n"
        for val in (sv.real, sv.imag, 0.0, 0.0, 0.0, 0.0):
            dataset += FMT_SPR(val)
        dataset += "\n"
    elif at in (4, 5):  # transient, frequency response
        for val in (2, 1, lc, ls):
            dataset += FMT_INT(val)
        dataset += "\n"
        dataset += FMT_SPR(sv) + "\n"
    else:               # general format
        for val in (2, 1, lc, ls):
            dataset += FMT_INT(val)
        dataset += "\n"
        dataset += FMT_SPR(sv) + "\n"

    if point_gids is None:
        node_gids = {i: i + 1 for i in range(len(point_data))}
    else:
        node_gids = {v: k for k, v in point_gids.items()}

    for i, row in enumerate(point_data):
        dataset += FMT_INT(node_gids[i]) + "\n"
        for j, val in enumerate(row):
            dataset += FMT_SPR(val)
            if (j + 1) % 6 == 0:
                dataset += "\n"
        if not dataset.endswith("\n"):
            dataset += "\n"

    dataset += DELIMITER + "\n"

    return dataset


def _write_point_data(point_data: np.ndarray,
                      point_gids: list=None,
                      header: dict=None) -> str:
    datasets = ""
    # TODO:
    if type(point_data) is np.ndarray:
        if header is None:
            header = {"analysis": "unknown",
                      "character": "unknown",
                      "type": "unknown",
                      "value type": "complex" if type(point_data[0,0]) is complex else "real",
                      "load case": 1,
                      "load step": 1,
                      "step value": 0.0}
        if "load case" not in header.keys():
            header["load case"] = 1
        if "load step" not in header.keys():
            header["load step"] = 1
        if "step value" not in header.keys():
            header["step value"] = 0.0
        datasets += _write_nodal_data(point_data, point_gids, header)
    elif type(point_data) in (list, tuple):
        if header is None:
            header = {"analysis": "unknown",
                      "character": "unknown",
                      "type": "scalar" if point_data.shape[1] == 1 else "unknown",
                      "value type": "complex" if type(point_data[0,0]) is complex else "real",
                      "load case": 1}
        for i in range(len(point_data)):
            if type(header) in (list, tuple):
                h = header[i]
            else:
                h = {k:v for k, v in header.items()} # copy values
            if "load case" not in h.keys():
                h["load case"] = 1
            if "load step" not in h.keys():
                h["load step"] = i + 1
            if "step value" not in h.keys():
                h["step value"] = 0.0
            datasets += _write_nodal_data(point_data[i], point_gids, h)
    elif type(point_data) is LoadCase:
        header = {"analysis": point_data.analysis,
                  "load case": point_data.id}
        for lstep in point_data.steps_by_id:
            header["load step"] = lstep.id
            header["step value"] = lstep.value
            for pdata in lstep.point_data:
                header["character"] = pdata.character
                header["type"] = pdata.type
                header["value type"] = pdata.value_type
                datasets += _write_nodal_data(pdata, point_gids, header)

    # for lcase in point_data.keys():
    #     for load_step in point_data[lcase].keys():
    #         datasets += _write_nodal_data(point_data[lcase][load_step], point_gids)

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
        "analysis":                   "static",
        "character":                 "vector3",
        "data type":            "displacement",
        "value type":                   "real",
        "value count":                       3,
        "load case":                       101,
        "load step":                         1,
        "step value":                      0.0,
    }

    dataset = _write_nodal_data(point_data, header)
    print(dataset)


    mesh = read("./res/hex_double_in.unv")
    print(mesh)
    # print(point_data)
    write("./res/hex_double_out.unv", mesh)
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

