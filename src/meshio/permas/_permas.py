"""
I/O for PERMAS dat files.
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


DFLT_COMP = 'KOMPO_1'
FMT_INT = '{0:11n}'
FMT_FLT = '{0:13.5E}'

permas_to_meshio_type = {
    "PLOT1":    "vertex",               # DEFAULT
    "PLOTL2":   "line",
    "FLA2":     "line",
    "BECOC":    "line",
    "BETAC":    "line",
    "BECOP":    "line",
    "BETOP":    "line",
    "BEAM2":    "line",
    "FSCPIPE2": "line",
    "BECOS":    "line",                 # DEFAULT
    "FLA3":     "line3",                # DEFAULT
    "PLOTL3":   "line3",                # DEFAULT
    "LOADA4":   "quad",
    "PLOTA4":   "quad",
    "QUAD4S":   "quad",
    "QUAMS4":   "quad",
    "SHELL4":   "quad",
    "QUAD4":    "quad",                 # DEFAULT
    "PLOTA8":   "quad8",
    "LOADA8":   "quad8",
    "QUAMS8":   "quad8",                # DEFAULT
    "PLOTA9":   "quad9",
    "LOADA9":   "quad9",
    "QUAMS9":   "quad9",                # DEFAULT
    "PLOTA3":   "triangle",
    "SHELL3":   "triangle",
    "TRIA3K":   "triangle",
    "TRIA3S":   "triangle",
    "TRIMS3":   "triangle",
    "TRIA3":    "triangle",             # DEFAULT
    "LOADA6":   "triangle6",
    "TRIMS6":   "triangle6",            # DEFAULT
    "HEXFO8":   "hexahedron",
    "HEXE8":    "hexahedron",           # DEFAULT
    "HEXE20":   "hexahedron20",         # DEFAULT
    "HEXE27":   "hexahedron27",         # DEFAULT
    "TET4":     "tetra",                # DEFAULT
    "TET10":    "tetra10",              # DEFAULT
    "PYRA5":    "pyramid",              # DEFAULT
    "PENTA6":   "wedge",                # DEFAULT
    "PENTA15":  "wedge15",              # DEFAULT
}
meshio_to_permas_type = {v: k for k, v in permas_to_meshio_type.items()}

meshio_to_permas_node_order = {
    'triangle6': [0, 3, 1, 4, 2, 5],
    'tetra10':   [0, 4, 1, 5, 2, 6, 7, 8, 9, 3],
    'quad9':     [0, 4, 1, 7, 8, 5, 3, 6, 2],
    'wedge15':   [0, 6, 1, 7, 2, 8, 9, 10, 11, 3, 12, 4, 13, 5, 14],
}
permas_to_meshio_node_order = {}
for etype in meshio_to_permas_node_order.keys():
    permas_to_meshio_node_order[etype] = {i: meshio_to_permas_node_order[etype][i] for i in
                                       range(len(meshio_to_permas_node_order[etype]))}
    permas_to_meshio_node_order[etype] = {v: k for k, v in
                                          permas_to_meshio_node_order[etype].items()}
    permas_to_meshio_node_order[etype] = [permas_to_meshio_node_order[etype][i] for i in
                                          sorted(permas_to_meshio_node_order[etype].keys())]


def read(filename):
    """Reads a PERMAS dat file."""
    with open_file(filename, "r") as f:
        out = read_buffer(f)
    return out


def read_buffer(f):
    # Initialize the optional data fields
    points = []
    point_gids = []
    cells = {}
    cell_gids = {}
    nsets = {}
    elsets = {}
    field_data = {}
    cell_data = {}
    point_data = {}

    _nsets = {}
    _nsets_numeric = {}
    _elsets = {}
    _elsets_numeric = {}

    while True:
        line, last_pos = _read_line(f)
        if line is None:      # EOF
            break
        elif line == "":      # empty or commented lines
            continue

        # keyword = line.strip("$").upper()
        keyword = line.upper()
        if keyword.startswith("$COOR"):
            params_map = get_param_map(keyword[1:])
            _points, _point_gids = _read_nodes(f)
            points.extend(_points)
            point_gids.extend(_point_gids)

            if "NSET" in params_map.keys():
                name = params_map["NSET"]
                if name not in _nsets:
                    _nsets.setdefault(name, [])
                    _nsets_numeric[name] = True
                _nsets[name].extend(point_gids)

        elif keyword.startswith("$ELEMENT"):
            params_map = get_param_map(keyword[1:])
            cell_type, idx, _cell_gids = _read_cells(f, keyword, params_map)
            if cell_type not in cells.keys():
                cells.setdefault(cell_type, [])
                cell_gids.setdefault(cell_type, [])
            cells[cell_type].extend(idx)
            cell_gids[cell_type].extend(_cell_gids)

            if "ESET" in params_map.keys():
                name = params_map["ESET"]
                if name not in _elsets:
                    _elsets[name] = []
                    _elsets_numeric[name] = True
                _elsets[name].extend(_cell_gids)

        elif keyword.startswith("$NSET"):
            params_map = get_param_map(keyword[1:], required_keys=["NAME"])
            setids, is_numeric = _read_set(f, params_map)
            name = params_map["NAME"]
            if name not in _nsets:
                _nsets[name] = []
                _nsets_numeric[name] = is_numeric
            _nsets[name].extend(setids)
            if not is_numeric:
                _nsets_numeric[name] = False

        elif keyword.startswith("$ESET"):
            params_map = get_param_map(keyword[1:], required_keys=["NAME"])
            setids, is_numeric = _read_set(f, params_map)
            name = params_map["NAME"]
            if name not in _elsets.keys():
                _elsets[name] = []
                _elsets_numeric[name] = is_numeric
            _elsets[name].extend(setids)
            if not is_numeric:
                _elsets_numeric[name] = False

        else:
            # There are just too many PERMAS keywords to explicitly skip them.
            # TODO:
            # if keyword.startswith("$"):
            #     print(f"Unsupported keyword {keyword.split(' ')[0]:s} found on line {last_pos+1:n}.")
            pass

    # prepare point gids
    point_gids = {point_gids[i]: i for i in range(len(point_gids))}

    # prepare points
    points = np.array(points, dtype=np.float64)

    # renumber cell nodes and cell_gids
    for cell_type in cells.keys():
        cell_count = 0
        for i in range(len(cells[cell_type])):
            cells[cell_type][i] = [point_gids[gid] for gid in cells[cell_type][i]]

        cell_gids[cell_type] = {cell_gids[cell_type][i]: i + cell_count
                                for i in range(len(cell_gids[cell_type]))}

        cell_count += len(cells[cell_type])

    cells = [CellBlock(etype, np.array(cells[etype], dtype=np.int32),
                       cell_gids=cell_gids[etype]) for etype in cells.keys()]

    nsets = _process_nsets(_nsets, _nsets_numeric, point_gids)
    elsets = _process_elsets(_elsets, _elsets_numeric, cells)

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data,
        point_sets=nsets, cell_sets=elsets, point_gids=point_gids
    )


def _set_name_to_id(setname, set, sets):
    _set = []
    is_numeric = True
    for nid in set:
        if type(nid) is int:
            _set.append(nid)
        elif nid.isnumeric():
            _set.append(int(nid))
        else:
            if nid in sets.keys():
                for iid in sets[nid]:
                    if type(iid) is int:
                        _set.append(iid)
                    elif type(iid) is str:
                        if iid.isnumeric():
                            _set.append(int(iid))
                        else:
                            _set.append(iid)
                            is_numeric = False
            else:
                raise ReadError(f"Node Set {nid} not found in file")
    return _set, is_numeric


def _process_nsets(nsets, nsets_numeric, point_gids):
    # replace set names by their contents
    while not all(nsets_numeric.values()):
        for nset in nsets.keys():
            if not nsets_numeric[nset]:
                nsets[nset], nsets_numeric[nset] = _set_name_to_id(nset, nsets[nset], nsets)

    # replace original ids by meshio ids
    msg = ""
    for nset in nsets.keys():
        nsets[nset] = list(set(nsets[nset]))
        swapped = [False] * len(nsets[nset])
        for i, nid in enumerate(nsets[nset]):
            nsets[nset][i] = point_gids[nid]
            swapped[i] = True
        if not all(swapped):
            msg += f"NSET NAME = {nset:s} IDs not found:\n"
            msg += ", ".join([nsets[nset][i] for i in range(len(nsets[nset])) if not swapped[i]]) + "\n"

    if msg != "":
        raise ReadError(msg)

    return nsets

def _process_elsets(elsets, elsets_numeric, cells):
    # replace set names by their contents
    while not all(elsets_numeric.values()):
        for elset in elsets.keys():
            if not elsets_numeric[nset]:
                elsets[elset], elsets_numeric[elset] = _set_name_to_id(elset, elsets[elset], elsets)

    # replace original ids by meshio ids
    msg = ""
    for elset in elsets.keys():
        elsets[elset] = list(set(elsets[elset]))
        swapped = [False] * len(elsets[elset])
        offset = 0
        for cell_block in cells:
            for i, eid in enumerate(elsets[elset]):
                if eid in cell_block.cell_gids.keys():
                    elsets[elset][i] = cell_block.cell_gids[eid] + offset
                    swapped[i] = True
            offset += len(cell_block.data)
        if not all(swapped):
            msg += f"ESET NAME = {elset:s} IDs not found:\n"
            msg += ", ".join([elsets[elset][i] for i in range(len(elsets[elset])) if not swapped[i]]) + "\n"

    if msg != "":
        raise ReadError(msg)

    return elsets


def _read_line(f: io.TextIOWrapper) -> (str, int):
    """
    Line reader in case of continuation lines in PERMAS

    Example
    -------
    >>> file = '! testing component

                $ENTER COMPONENT
                  ! & NAME = KOMPO_2
                  & NAME = KOMPO_1  ! correct name

                  & DOFTYPE = DISP
                  $STRUCTURE'
    >>> line, last_pos =_read_line(file)
    line = ""
    last_pos = 0
    >>> line, last_pos =_read_line(file)
    line = ""
    last_pos = 1
    >>> line, last_pos =_read_line(file)
    line = '$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP'
    last_pos = 2
    >>> line, last_pos =_read_line(file)
    line = '$STRUCTURE'
    last_pos = 7
    """
    last_pos = f.tell()                       # previous line in file
    line = f.readline()                       # read current line
    if not line:                              # EOF
        return None, last_pos                 # return None, previous position
    line = _strip_line(line)                  # strip witespaces and comments
    if line == "":                            # in case of empty or commented line
        return line, last_pos
    while True:                               # cycle through next lines
        last_pos1 = f.tell()                  # last position in file
        line1 = f.readline()                  # read next line
        if not line1:                         # next line is EOF
            f.seek(last_pos1)                 # revert back
            break
        line1 = _strip_line(line1)            # strip whitespaces and comments
        if line1.startswith("&"):             # line is continuation
            last_pos = last_pos1              # update last position
            line += " " + line1[1:].strip()   # concatenate lines
        elif line1 == "":                     # next line is empty or commented
            last_pos = last_pos1              # update last position
        else:                                 # next line is normal
            f.seek(last_pos1)                 # revert back
            break

    line = _strip_line(line)                  # one more strip in case of commented or empty
                                              # lines in the middle of continuation block
    return line, last_pos


def _strip_line(line: str) -> str:
    """
    Strips line of whitespaces and comments, if whole line is commented returns
    empty string.
    """
    line = line.strip()                       # strip whitespace characters
    while "  " in line:                       # replace multiple spaces with single ones
        line = line.replace("  ", " ")
    if line.startswith("!") or line == " ":   # discard commented line or empty line
        line = ""
    if "!" in line:                           # discard comment at EOL
        line = line.split("!")[0].strip(" ")
    return line


def _read_nodes(f):
    points = []
    point_gids = []
    # index = 0
    while True:
        line, last_pos = _read_line(f)
        if line is None:             # EOF
            break
        elif line == "":             # empty or commented line
            continue
        elif line.startswith("$"):   # next keyword
            break

        entries = line.split(" ")
        gid, x = entries[0], entries[1:]
        point_gids.append(int(gid))
        points.append([float(xx) for xx in x])

    f.seek(last_pos)
    return points, point_gids


def _read_cells(f, line0, params_map):
    if "TYPE" not in params_map.keys():
        raise ReadError(line0)
    etype = params_map["TYPE"]
    if etype not in permas_to_meshio_type:
        raise ReadError(f"Element type not available: {etype}")
    cell_type = permas_to_meshio_type[etype]
    cells, idx = [], []
    cell_gids = []
    while True:
        line, last_pos = _read_line(f)
        if line is None:             # EOF
            break
        elif line == "":             # empty or commented line
            continue
        elif line.startswith("$"):   # next keyword
            break

        entries = [int(k) for k in filter(None, line.split(" "))]
        idx = entries[1:]
        if cell_type in permas_to_meshio_node_order.keys():
            idx = [idx[i] for i in permas_to_meshio_node_order[cell_type]]

        cells.append(idx)
        cell_gids.append(entries[0])

    f.seek(last_pos)
    return cell_type, cells, cell_gids


def _read_set(f, params_map):
    set_ids = []
    # TODO:
    # setids can be also set names
    is_numeric = True
    while True:
        last_pos = f.tell()
        line = f.readline()
        line = _strip_line(line)
        if line is None:             # EOF
            break
        elif line == "":             # empty or commented line
            continue
        elif line.startswith("$"):   # next keyword
            break
        # cannot convert to int in case of set names included
        for k in line.split(" "):
            if k.isnumeric():
                set_ids.append(int(k))
            else:
                is_numeric = False
                set_ids.append(k)

        # set_ids += [k for k in line.split(" ")]
    f.seek(last_pos)

    # TODO:
    # RULE = [ITEM/ALL/BOOLEAN/RANGE]
    if "RULE" in params_map and params_map["RULE"] != "ITEM":
        raise NotImplementedError("Reading other types of ses apart from RULE = ITEM " +
                                  f"is not implemented (RULE = {params_map['RULE']}).")
    # if "generate" in params_map:
    #     if len(set_ids) != 3:
    #         raise ReadError(set_ids)
    #     set_ids = np.arange(set_ids[0], set_ids[1], set_ids[2])
    # else:
    #     try:
    #         set_ids = np.unique(np.array(set_ids, dtype="int32"))
    #     except ValueError:
    #         raise
    # try:
    #     set_ids = np.unique(np.array(set_ids, dtype="int32"))
    # except ValueError:
    #     raise
    return set_ids, is_numeric


def get_param_map(word, required_keys=None):
    """
    get the optional arguments on a line

    Example
    -------
    >>> iline = 0
    >>> word = '$NSET NAME = OF_ALLE'
    >>> params = get_param_map(iline, word, required_keys=['instance'])
    params = {
        'NSET' : None,
        'NAME' : 'OF_ALLE',
    }
    >>> word = '$ENTER COMPONENT NAME = KOMPO_1 DOFTYPE = DISP PRES TEMP MATH'
    >>> params = get_param_map(iline, word)
    params = {
        'ENTER' : None,
        'COMPONENT' : None,
        'NAME' : 'KOMPO_1',
        'DOFTYPE': ['DISP', 'PRES', 'TEMP', 'MATH']
    }
    """
    word = word.lstrip("$").upper().strip() # strip whitespaces and ledt $
    word = word.replace("=", " = ")         # create spaces around =
    while "  " in word:                     # replace double spaces with single one
        word = word.replace("  ", " ")
    word = word.replace(" =", "=")          # delete space before =

    sword = word.split(" ")                 # split using spaces

    i = 0
    param_map = {}
    key = ""
    while i < len(sword):
        if sword[i].endswith("="):
            key = sword[i][:-1]
            param_map.setdefault(key, [])
        elif key == "":
            param_map[sword[i]] = []
        else:
            param_map[key].append(sword[i])
        i += 1
    # raise ReadError(sword)

    for key, values in param_map.items():
        if len(values) == 0:
            param_map[key] = None
        elif len(values) == 1:
            param_map[key] = values[0]

    if required_keys is None:
        required_keys = []
    msg = ""
    for key in required_keys:
        if key not in param_map:
            msg += f"{key} not found in {word}\n"
    if msg:
        raise RuntimeError(msg)

    return param_map


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
                if cell_block.type in meshio_to_permas_node_order.keys():
                    mylist = [mylist[i] for i in meshio_to_permas_node_order[cell_block.type]]
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
    "permas", [".post", ".post.gz", ".dato", ".dato.gz"], read, {"permas": write}
)

if __name__ == "__main__":
    mesh = read("./test_in.dat")
    print(mesh)
    write("./test_out.dat", mesh)

