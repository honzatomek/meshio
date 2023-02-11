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


DELIMITER = f'{-1:6n}'
FMT_DTS = "{0:>6n}"
FMT_INT = "{0:10n}"
FMT_FLT = "{0:13.5E}"
FMT_DBL = "{0:25.16E}"

unv_to_meshio_dataset = {
    15: "NODESP",
  2411: "NODEDP",
  2412: "ELEMENT",
    55: "NODAL DATA"
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

meshio_node_order = {
    "triangle6": [0, 3, 1, 4, 2, 5],
    "tetra10":   [0, 4, 1, 5, 2, 6, 7, 8, 9, 3],
    "quad9":     [0, 4, 1, 7, 8, 5, 3, 6, 2],
    "wedge15":   [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14],
}
unv_node_order = {}
for etype in meshio_node_order.keys():
    unv_node_order[etype] = {i: meshio_node_order[etype][i] for i in range(len(meshio_node_order[etype]))}
    unv_node_order[etype] = {v: k for k, v in unv_node_order[etype].items()}
    unv_node_order[etype] = [unv_node_order[etype][i] for i in sorted(unv_node_order[etype].keys())]


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
            if unv_to_meshio_dataset[dataset] == "NODESP":
                _points, _point_gids = _read_sp_nodes(f)
                points.extend(_points)
                point_gids.extend(_point_gids)

            elif unv_to_meshio_dataset[dataset] == "NODEDP":
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
                _read_dataset(f)

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

        pids = []
        while len(pids) < numnodes:
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
            pids.extend([int(line[j*10:(j+1)*10].strip()) for j in range(numnodes)])

        if FEid not in unv_to_meshio_type.keys():
            err += f"Wrong type of element {cid:n} at position {last_pos:n}.\n"
            continue

        cell_type = unv_to_meshio_type[FEid]
        if cell_type not in cells.keys():
            cells.setdefault(cell_type, [])
            cell_gids.setdefault(cell_type, [])

        if cell_type in meshio_node_order.keys():
            cells[cell_type].append([pids[meshio_node_order[etype][i]] for i in range(len(pids))])
        else:
            cells[cell_type].append([pids[i] for i in range(len(pids))])
        cell_gids[cell_type].append(cid)

    if err != "":
        raise ReadError(err)

    return cells, cell_gids


def _read_dataset(f):
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


def _write_nodes(points: list, offset_len: int=6,
                 fmt_nid: str=FMT_INT, fmt_coor=FMT_FLT, node_gids: dict=None) -> str:
    offset = " " * offset_len
    lines = []
    for id, coors in enumerate(points):
        nid = id + 1 if node_gids is None else node_gids[id]
        lines.append(offset + fmt_nid.format(nid) + "".join([fmt_coor.format(x) for x in coors]))
    return "\n".join(lines) + "\n"


def _write_element(eid: int, nodes: list, maxlinelen: int=80, offset_len: int=6,
                   fmt_eid: str=FMT_INT, fmt_nid: str=FMT_INT,
                   node_gids: list=None, element_gids: list=None, row: int=None) -> str:

    if element_gids is not None:
        eid = element_gids[row]

    if node_gids is not None:
        nodes = [node_gids[node] for node in nodes]
    else:
        nodes = [node + 1 for node in nodes]

    offset = " " * offset_len
    continuation = ("{0:<" + str(len(fmt_eid.format(1))) + "}").format("&")
    lines = []
    nids_strs = (fmt_nid.format(nid) for nid in nodes)
    ncount = len(nodes)
    line = offset + fmt_eid.format(eid)
    for i, n in enumerate(nids_strs):
        line += n
        if len(line) + len(n) > maxlinelen or i == ncount - 1:
            lines.append(line)
            line = offset + continuation

    return "\n".join(lines) + "\n"


def _write_set(nodes: list, maxlinelen: int=80, offset_len: int=6,
               fmt_nid: str=FMT_INT, node_gids: list=None, id_offset=1) -> str:
    lines = []
    ncount = len(nodes)
    if node_gids is not None:
        nodes = [node_gids[node] for node in nodes]
    else:
        nodes = [node + id_offset for node in nodes]
    offset = " " * offset_len
    nids_strs = (fmt_nid.format(nid) for nid in nodes)
    line = offset
    for i, n in enumerate(nids_strs):
        line += n
        if len(line) + len(n) > maxlinelen or i == ncount - 1:
            lines.append(line)
            line = offset

    return "\n".join(lines) + "\n"


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
        node_gids = None if mesh.point_gids is None else {v: k for k, v in mesh.point_gids.items()}
        point_gids = mesh.point_gids
        f.write("! PERMAS DataFile Version 18.0\n")
        f.write(f"! written by meshio v{__version__}\n")
        f.write(f"$ENTER COMPONENT NAME = {DFLT_COMP:s}\n")
        f.write("  $STRUCTURE\n")
        f.write("    $COOR\n")
        f.write(_write_nodes(points, 6, FMT_INT, FMT_FLT, node_gids))
        eid = 0

        for cell_block in mesh.cells:
            node_idcs = cell_block.data
            cell_gids = cell_block.cell_gids
            element_gids = None if cell_gids is None else {v: k for k, v in cell_gids.items()}
            f.write("!\n")
            f.write("    $ELEMENT TYPE = " + meshio_to_permas_type[cell_block.type] + "\n")
            for i, row in enumerate(node_idcs):
                eid += 1
                mylist = row.tolist()
                if cell_block.type in meshio_node_order.keys():
                    mylist = [mylist[i] for i in meshio_node_order[cell_block.type]]
                f.write(_write_element(eid, mylist, 80, 6, FMT_INT, FMT_INT,
                                       node_gids, element_gids, i))

        f.write("!\n")

        for point_set, points in mesh.point_sets.items():
            f.write(f"  $NSET NAME = {point_set:s}\n")
            f.write(_write_set(points, 80, 6, FMT_INT, node_gids))
            f.write("!\n")

        for cell_set, cellids in mesh.cell_sets.items():
            eset = []
            f.write(f"  $ESET NAME = {cell_set:s}\n")
            if element_gids is None:
                f.write(_write_set(points, 80, 6, FMT_INT, cellids))
            else:
                offset = 0
                for cell_block in mesh.cells:
                    cell_gids = cell_block.cell_gids
                    if cell_gids is None:
                        cell_gids = {i + offset: i for i in range(len(cell_block.data))}
                    else:
                        element_gids = {v + offset: k for k, v in cell_gids.items()}
                    for cellid in cellids:
                        if cellid in element_gids.keys():
                            eset.append(element_gids[cellid])
                    offset += len(mesh.cells)
                f.write(_write_set(eset, 80, 6, FMT_INT, None, id_offset=0))
            f.write("!\n")

        f.write("  $END STRUCTURE\n")
        f.write("!\n")
        f.write("$EXIT COMPONENT\n")
        f.write("$FIN\n")


register_format(
    "unv", [".unv", ".unv.gz"], read, {"unv": write}
)

if __name__ == "__main__":
    mesh = read("./res/hex_double.unv")
    print(mesh)
    from meshio import permas
    permas.write("./res/test_hex_double.dat", mesh)

