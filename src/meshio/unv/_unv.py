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
    15: "NODE1P",
  2411: "NODE2P",
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
                _read_dataset(f)

            # TODO:
            # nsets, elsets

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
            for i in range(numnodes):
                dataset += f"{nodes[i]:10n}"
                if (i + 1) % 8 == 0:
                    dataset += "\n"
            if not dataset.endswith("\n"):
                dataset += "\n"
            cid += 1

    dataset += DELIMITER + "\n"

    return dataset


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
    mesh = read("./res/hex_double_in.unv")
    print(mesh)
    write("./res/hex_double_out.unv", mesh)

