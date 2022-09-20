"""
I/O for unv dat files.
"""
import numpy as np

from ..__about__ import __version__
from .._common import warn
from .._exceptions import ReadError
from .._files import open_file
from .._helpers import register_format
from .._mesh import CellBlock, Mesh

unv_to_meshio_type = {
    "11": "line",
    "21": "line",
    "22": "line3",
    "24": "line3",
    "44": "quad",
    "45": "quad8",
    "41": "triangle",
    "42": "triangle6",
    "115": "hexahedron",
    "116": "hexahedron27",
    "111": "tetra",
    "118": "tetra10",
    "112": "wedge",
}
meshio_to_unv_type = {v: k for k, v in unv_to_meshio_type.items()}


def read(filename):
    """Reads a unv dat file."""
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
        if line == f'{-1:6n}':
          line = f.readline()

        keyword = int(line.strip())
        if keyword in [15, 781, 2411]:
            p, pg = _read_nodes(f, keyword)
            points.extend(p)
            point_gids.extend(pg)
        elif keyword in [780, 2412]
            cell_types = _read_cells(f, keyword)
            for key, idx in cell_types.items():
                cells.append(CellBlock(key, idx))
        elif keyword.startswith("NSET"):
            params_map = get_param_map(keyword, required_keys=["NSET"])
            setids = read_set(f, params_map)
            name = params_map["NSET"]
            if name not in nsets:
                nsets[name] = []
            nsets[name].append(setids)
        elif keyword.startswith("ESET"):
            params_map = get_param_map(keyword, required_keys=["ESET"])
            setids = read_set(f, params_map)
            name = params_map["ESET"]
            if name not in elsets:
                elsets[name] = []
            elsets[name].append(setids)
        else:
            # There are just too many unv keywords to explicitly skip them.
            pass

        for cb in cells:
            for i, idx in enumerate(cb.data):
                idx = [point_gids[int(k)] for k in idx]
                cb.data[i] = np.array(idx)


    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def _read_nodes(f, keyword):
    points = []
    point_gids = {}
    index = 0
    while True:
        last_pos = f.tell()
        line = f.readline()
        if line == f'{-1:6n}':
            break
        if keyword == 15:
          entries = line.strip().split(" ")
          gid, x = entries[0], entries[4:]
          point_gids[int(gid)] = index
          points.append([float(xx) for xx in x])
          index += 1
        elif keyword in [781, 2411]:
          entries = line.strip().split(" ")
          gid = entries[0]
          point_gids[int(gid)] = index
          line = f.readline()
          x = line.strip().split(" ")
          points.append([float(xx.lower().replace('d', 'e')) for xx in x])
        else:
          raise ReadError(keyword)
        index += 1

    # f.seek(last_pos)
    return np.array(points, dtype=float), point_gids


def _read_cells(f):
    cells = {}
    while True:
        line = f.readline()
        if line == f'{-1:6n}':
            break
        sline = line.split(" ")
        etype = int(sline.split(" ")[1].strip())
        if etype not in unv_to_meshio_type:
            raise ReadError(f"Element type not available: {etype}")
        nnode = int(sline.split(" ")[-1].strip())
        cell_type = unv_to_meshio_type[etype]
        line = f.readline():
        if etype in [21, 22, 24]:
            line = f.readline():
        nodes = line.strip().split()
        if len(nodes) < nnode:
            nodes.append(line.strip().split())
        cells[cell_type].append([int(n) for n in nodes])
    for cell_type in cells.keys():
      cells[cell_type] = np.array(cells[cell_type])
    return cells


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> iline = 0
    >>> word = 'elset,instance=dummy2,generate'
    >>> params = get_param_map(iline, word, required_keys=['instance'])
    params = {
        'elset' : None,
        'instance' : 'dummy2,
        'generate' : None,
    }
    """
    if required_keys is None:
        required_keys = []
    words = word.split(",")
    param_map = {}
    for wordi in words:
        if "=" not in wordi:
            key = wordi.strip()
            value = None
        else:
            sword = wordi.split("=")
            if len(sword) != 2:
                raise ReadError(sword)
            key = sword[0].strip()
            value = sword[1].strip()
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
        set_ids += [int(k) for k in line.strip().strip(" ").split(" ")]
    f.seek(last_pos)

    if "generate" in params_map:
        if len(set_ids) != 3:
            raise ReadError(set_ids)
        set_ids = np.arange(set_ids[0], set_ids[1], set_ids[2])
    else:
        try:
            set_ids = np.unique(np.array(set_ids, dtype="int32"))
        except ValueError:
            raise
    return set_ids


def write(filename, mesh):
    if mesh.points.shape[1] == 2:
        warn(
            "unv requires 3D points, but 2D points given. "
            "Appending 0 third component."
        )
        points = np.column_stack([mesh.points, np.zeros_like(mesh.points[:, 0])])
    else:
        points = mesh.points

    with open_file(filename, "wt") as f:
        f.write("!unv DataFile Version 18.0\n")
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
            f.write("$ELEMENT TYPE=" + meshio_to_unv_type[cell_block.type] + "\n")
            if cell_block.type == "tetra10":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in tet10_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "triangle6":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in tria6_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "quad9":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in quad9_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            elif cell_block.type == "wedge15":
                for row in node_idcs:
                    eid += 1
                    mylist = row.tolist()
                    mylist = [mylist[i] for i in wedge15_order]
                    nids_strs = (str(nid + 1) for nid in mylist)
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
            else:
                for row in node_idcs:
                    eid += 1
                    nids_strs = (str(nid + 1) for nid in row.tolist())
                    f.write(str(eid) + " " + " ".join(nids_strs) + "\n")
        f.write("$END STRUCTURE\n")
        f.write("$EXIT COMPONENT\n")
        f.write("$FIN\n")


register_format(
    "unv", [".unv", ".uff"], read, {"unv": write}
)
