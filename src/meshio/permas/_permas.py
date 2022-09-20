"""
I/O for PERMAS dat files.
"""
import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

permas_to_meshio_type = {
    "PLOT1": "vertex",
    "PLOTL2": "line",
    "FLA2": "line",
    "FLA3": "line3",
    "PLOTL3": "line3",
    "BECOS": "line",
    "BECOC": "line",
    "BETAC": "line",
    "BECOP": "line",
    "BETOP": "line",
    "FSCPIPE2": "line",
    "BEAM2": "line",
    "LOADA4": "quad",
    "PLOTA4": "quad",
    "QUAD4S": "quad",
    "QUAMS4": "quad",
    "SHELL4": "quad",
    "QUAD4": "quad",
    "PLOTA8": "quad8",
    "LOADA8": "quad8",
    "QUAMS8": "quad8",
    "PLOTA9": "quad9",
    "LOADA9": "quad9",
    "QUAMS9": "quad9",
    "PLOTA3": "triangle",
    "SHELL3": "triangle",
    "TRIA3K": "triangle",
    "TRIA3S": "triangle",
    "TRIMS3": "triangle",
    "TRIA3": "triangle",
    "LOADA6": "triangle6",
    "TRIMS6": "triangle6",
    "TRIA6": "triangle6",
    "HEXFO8": "hexahedron",
    "HEXE8": "hexahedron",
    "HEXE20": "hexahedron20",
    "HEXE27": "hexahedron27",
    "TET4": "tetra",
    "TET10": "tetra10",
    "PYRA5": "pyramid",
    "PENTA6": "wedge",
    "PENTA15": "wedge15",
}
meshio_to_permas_type = {v: k for k, v in permas_to_meshio_type.items()}

element_node_order = {
    "triangle6": [0, 3, 1, 4, 2, 5],
    "tetra10": [0, 4, 1, 5, 2, 6, 7, 8, 9, 3],
    "quad9": [0, 4, 1, 7, 8, 5, 3, 6, 2],
    "wedge15": [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14]
}


def read(filename):
    """Reads a PERMAS dat file."""
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Initialize the optional data fields
    cells = []
    nsets = {}
    elsets = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    while True:
        line = f.readline()
        if not line:
            # EOF
            break

        # Comments
        if line.startswith("!"):
            continue

        keyword = line.strip("$").upper()
        if keyword.startswith("COOR"):
            params_map = get_param_map(keyword, required_keys=["COOR"])
            points, point_gids = _read_nodes(f)
        elif keyword.startswith("ELEMENT"):
            params_map = get_param_map(keyword, required_keys=["ELEMENT", "TYPE"])
            key, idx = _read_cells(f, keyword, point_gids)
            cells.append(CellBlock(key, idx))
        elif keyword.startswith("NSET"):
            params_map = get_param_map(keyword, required_keys=["NSET", "NAME"])
            setids = read_set(f, params_map)
            name = params_map["NAME"]
            if name not in nsets:
                nsets[name] = []
            nsets[name].append(setids)
        elif keyword.startswith("ESET"):
            params_map = get_param_map(keyword, required_keys=["ESET", "NAME"])
            setids = read_set(f, params_map)
            name = params_map["NAME"]
            if name not in elsets:
                elsets[name] = []
            elsets[name].append(setids)
        else:
            # There are just too many PERMAS keywords to explicitly skip them.
            pass

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data, point_sets=nsets, cell_sets=elsets
    )


def _strip_comments(line):
    if line.strip().startswith("!"):
        line = ""
    elif "!" in line:
        line = line.split("!")[0]
    return line


def _strip_spaces(line):
    line = line.strip()
    while "  " in line:
        line = line.replace("  ", " ")
    if "=" in line:
        line = line.replace(" =", "=").replace("= ", "=")
    return line


def _strip_all(line):
    return _strip_spaces(_strip_comments(line))


def _read_nodes(f):
    points = []
    point_gids = {}
    index = 0
    while True:
        last_pos = f.tell()
        line = f.readline()
        if not line:
            # EOF
            break
        line = _strip_all(line)
        if line == "":
            continue
        elif line.startswith("$"):
            break
        entries = line.split(" ")
        gid, x = entries[0], entries[1:]
        point_gids[int(gid)] = index
        points.append([float(xx) for xx in x])
        index += 1

    f.seek(last_pos)
    return np.array(points, dtype=float), point_gids


def _read_cells(f, line0, point_gids):
    params_map = get_param_map(_strip_all(line0), ["ELEMENT", "TYPE"])
    if params_map["TYPE"] is None:
        raise ReadError(line0)
    etype = params_map["TYPE"]
    if etype not in permas_to_meshio_type:
        raise ReadError(f"Element type not available: {etype}")
    cell_type = permas_to_meshio_type[etype]
    cells, idx = [], []
    while True:
        last_pos = f.tell()
        line = f.readline()
        if not line:
            # EOF
            break
        line = _strip_all(line)
        if line == "":
            continue
        elif line.startswith("$"):
            break
        # the first item is just a running index
        idx += [point_gids[int(k)] for k in filter(None, line.split(" ")[1:])]
        cells.append(idx)
        idx = []
    f.seek(last_pos)
    return cell_type, np.array(cells)


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> word = 'ESET NAME=DUMMY RULE=RANGE'
    >>> params = get_param_map(word, required_keys=['NAME'])
    params = {
        'ESET' : None,
        'NAME' : 'DUMMY',
        'RULE' : 'RANGE',
    }
    """
    if required_keys is None:
        required_keys = []
    word = _strip_all(word)
    words = word.split(" ")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip().upper()
            value = sword[1].strip().strip("'").strip('"').upper()
        param_map[key] = value

    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += f"{key} not found in {word}\n"
    if msg:
        raise RuntimeError(msg)
    return param_map


def read_set(f, params_map):
    set_ids = []
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line.startswith("$"):
            break
        if "RULE" not in params_map or params_map["RULE"] == "ITEM":
            set_ids += [int(k) for k in line.strip().strip(" ").split(" ")]
        elif params_map["RULE"] == "RANGE":
            for bounds in line.strip().strip(" ").split(":"):
                set_ids.append([int(b) for b in bounds.split(" ")])
        else:
            raise NotImplementedError(f"{params_map.keys()[0]:s} RULE = {params_map['RULE']:s} not implemented!")
    f.seek(last_pos)

    if "RULE" in params_map and params_map["RULE"] == "RANGE":
        bound_ids = np.array(dtype="int32")
        for bounds in set_ids:
            if len(bounds) != 2:
                raise ReadError(str(bounds))
            set_ids += list(range(bounds[0], bounds[1] + 1))

    try:
        set_ids = np.unique(np.array(set_ids, dtype="int32"))
    except ValueError:
        raise
    return set_ids


def _set_name(name):
    if ' ' in name:
        return "'" + name + "'"
    else:
        return name


def _rows_of_ids(ids, num_per_line = 8):
    idlen = len(ids)
    for i in range(0, len(ids), num_per_line):
        yield " ".join([f"{i:n}" for id in ids[i:min(i+num_per_line,idlen)]])


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        warn(
            "PERMAS requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    with open_file(filename, "wt") as f:
        f.write("!PERMAS DataFile Version 18.0\n")
        f.write(f"!written by meshio v{__version__}\n")
        f.write("$ENTER COMPONENT NAME=DFLT_COMP\n")
        f.write("$STRUCTURE\n")
        f.write("$COOR\n")
        for k, x in enumerate(points):
            f.write(f"{k + 1} {x[0]} {x[1]} {x[2]}\n")
        eid = 0
        tria6_order = [0, 3, 1, 4, 2, 5]
        tet10_order = [0, 4, 1, 5, 2, 6, 7, 8, 9, 3]
        quad9_order = [0, 4, 1, 7, 8, 5, 3, 6, 2]
        wedge15_order = [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14]
        for cell_block in mesh.cells:
            node_idcs = cell_block.data
            f.write("!\n")
            f.write("$ELEMENT TYPE=" + meshio_to_permas_type[cell_block.type] + "\n")
            if cell_block.type in element_node_order.keys():
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in element_node_order[cell_block.type]]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            else:
                for row in node_idcs:
                    eid += 1
                    nids_strs = (str(nid + 1) for nid in row.tolist())
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
        for nset, nids in mesh.point_sets.items():
            f.write("$NSET NAME='" + _set_name(nset) + '\n')
            for row in _rows_of_ids(nids, 8):
                f.write(row + "\n")
        for eset, sets in mesh.cell_sets.items():
            if type(sets) is list:
                f.write("$ESET NAME='" + _set_name(eset) + '\n')
                for row in _rows_of_ids(sets, 8):
                    f.write(row + '\n')
            elif type(sets) is dict:
                f.write("$ESET NAME='" + _set_name(eset) + '\n')
                for etype, eids in sets.items():
                    for row in _rows_of_ids(eids, 8):
                        f.write(row + '\n')
            else:
                pass
        f.write("$END STRUCTURE\n")
        f.write("$EXIT COMPONENT\n")
        f.write("$FIN\n")


register_format(
    "permas", [".post", ".post.gz", ".dato", ".dato.gz"], read, {"permas": write}
)
