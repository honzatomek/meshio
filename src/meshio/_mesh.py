from __future__ import annotations

import copy

import numpy as np
from numpy.typing import ArrayLike

from ._common import num_nodes_per_cell, warn

topological_dimension = {
    "line": 1,
    "polygon": 2,
    "triangle": 2,
    "quad": 2,
    "tetra": 3,
    "hexahedron": 3,
    "wedge": 3,
    "pyramid": 3,
    "line3": 1,
    "triangle6": 2,
    "quad9": 2,
    "tetra10": 3,
    "hexahedron27": 3,
    "wedge18": 3,
    "pyramid14": 3,
    "vertex": 0,
    "quad8": 2,
    "hexahedron20": 3,
    "triangle10": 2,
    "triangle15": 2,
    "triangle21": 2,
    "line4": 1,
    "line5": 1,
    "line6": 1,
    "tetra20": 3,
    "tetra35": 3,
    "tetra56": 3,
    "quad16": 2,
    "quad25": 2,
    "quad36": 2,
    "triangle28": 2,
    "triangle36": 2,
    "triangle45": 2,
    "triangle55": 2,
    "triangle66": 2,
    "quad49": 2,
    "quad64": 2,
    "quad81": 2,
    "quad100": 2,
    "quad121": 2,
    "line7": 1,
    "line8": 1,
    "line9": 1,
    "line10": 1,
    "line11": 1,
    "tetra84": 3,
    "tetra120": 3,
    "tetra165": 3,
    "tetra220": 3,
    "tetra286": 3,
    "wedge40": 3,
    "wedge75": 3,
    "hexahedron64": 3,
    "hexahedron125": 3,
    "hexahedron216": 3,
    "hexahedron343": 3,
    "hexahedron512": 3,
    "hexahedron729": 3,
    "hexahedron1000": 3,
    "wedge126": 3,
    "wedge196": 3,
    "wedge288": 3,
    "wedge405": 3,
    "wedge550": 3,
    "VTK_LAGRANGE_CURVE": 1,
    "VTK_LAGRANGE_TRIANGLE": 2,
    "VTK_LAGRANGE_QUADRILATERAL": 2,
    "VTK_LAGRANGE_TETRAHEDRON": 3,
    "VTK_LAGRANGE_HEXAHEDRON": 3,
    "VTK_LAGRANGE_WEDGE": 3,
    "VTK_LAGRANGE_PYRAMID": 3,
}

meshio_analysis = (
    "unknown",
    "static",
    "modal",
    "modal complex",
    "transient",
    "frequency response",
    "buckling",
    "nlstatic"
)


meshio_data_types = (
    "unknown",
    "stress",
    "strain",
    "force",
    "temperature",
    "heat flux",
    "strain energy",
    "displacement",
    "reaction",
    "kinetic energy",
    "velocity",
    "acceleration",
    "strain energy density",
    "kinetic energy density",
    "pressure",
    "heat",
    "check",
    "pressure coefficient",
    "ply stress",
    "ply strain",
    "cell scalar",
    "cell scalar",
    "reaction heat flow",
    "stress error density",
    "stress variation",
    "shell and plate elem stress resultant",
    "length",
    "area",
    "volume",
    "mass",
    "constraint forces",
    "plastic strain",
    "creep strain",
    "strain energy error",
    "dynamic stress at nodes",
    "cell unknown",
    "cell scalar",
    "cell vector3",
    "cell vector6",
    "cell symmetric tensor",
    "cell global tensor",
    "cell shell and plate resultant",
)

meshio_data_characteristic = {
    "unknown":           ["X"],
    "scalar":            ["X"],
    "vector3":           ["X", "Y", "Z"],
    "vector6":           ["Fx", "Fy", "Fz", "Rx", "Ry", "Rz"],
    "symmetric tensor":  ["Sxx", "Sxy", "Syy", "Sxz", "Syz", "Szz"],
    "global tensor":     ["Sxx", "Syx", "Szx", "Sxy", "Syy", "Szy", "Sxz", "Syz", "Szz"],
    "stress resultant":  ["N", "Qy", "Qz", "Mx", "My", "Mz"],
}


class CellBlock:
    def __init__(
        self,
        cell_type: str,
        data: list | np.ndarray,
        tags: list[str] | None = None,
        cell_gids: dict = None
    ):
        self.type = cell_type
        self.data = data
        self.cell_gids = cell_gids   # original element IDs

        if cell_type.startswith("polyhedron"):
            self.dim = 3
        else:
            self.data = np.asarray(self.data)
            self.dim = topological_dimension[cell_type]

        self.tags = [] if tags is None else tags

    def __repr__(self):
        items = [
            "meshio CellBlock",
            f"type: {self.type}",
            f"num cells: {len(self.data)}",
            f"tags: {self.tags}",
        ]
        return "<" + ", ".join(items) + ">"

    def __len__(self):
        return len(self.data)


class LoadCase:
    def __init__(self, load_steps: list[LoadStep], analysis: str="static",
                 load_case: int=101):
        """
        load case container

        In:
            analysis   - analysis type (unknown, static, modal, modal complex,
                         transient, frequency response, buckling, nlstatic)
            load_steps - a list of LoadStep containers, will be sorted based
                         on LoadStep.step property
            load_case  - load case ID
        """
        self.id = load_case
        self.analysis = analysis
        self.steps = load_steps


    def __len__(self) -> int:
        return len(self._steps)


    def __repr__(self) -> str:
        lines = ["<meshio loadcase object>",
                 f"  Analysis type: {self._analysis:s}",
                 f"  Load case ID: {self._case:n}",
                 f"  Number of load steps: {self.__len__():n}",
                 f"  Load value: {self.value:n}",
                 f"  Number of point data: {len(self.point_data)}",
                 f"  Number of cell data: {len(self.cell_data)}"]
        return "\n".join(lines)


    def __iter__(self) -> tuple[int, LoadStep]:
        for lstep in self._steps:
            yield lstep.step


    def __getitem__(self, load_step: int) -> LoadStep:
        return self._steps[self._stepids[load_step]]


    @property
    def id(self) -> int:
        return self._id


    @id.setter
    def id(self, step_id: int):
        self._id = int(step_id)


    @property
    def steps(self) -> list[LoadStep]:
        return self._steps


    @steps.setter
    def steps(self, load_steps: list[LoadStep]):
        self._steps = load_steps.sorted(key=lambda lstep: lstep.step)
        self._stepids = {lstep.id: i for lstep in self._steps}


    @property
    def steps_by_step_id(self) -> list[LoadStep]:
        return self._steps


    @property
    def steps_by_value(self) -> list[LoadStep]:
        return self._steps.sorted(key=lambda lstep: lstep.value)


    @property
    def case(self) -> int:
        return self._case


    @case.setter
    def case(self, load_case: int):
        self._case = int(load_case)

    @property
    def analysis(self) -> str:
        return self._analysis

    @analysis.setter
    def analysis(self, analysis: str):
        if analysis not in meshio_analysis:
            self._analysis = "unknown"
        else:
            self._analysis = analysis



class LoadStep:
    def __init__(self, point_data: list[Data]=None, cell_data: list[Data]=None,
                 step_id: int=1, value: [float | complex]= 0.0):
        """
        a container of data for one load step, can contain multiple
        vectors, scalars, tensors...

        In:
            point_data - a list of all possible scalars, vectors
                         and such
            cell_data  - a list of all possible scalars, vectors
                         and such
            step_id    - load step ID (e.g. mode shape number,
                         time step ID for transient analysis,
                         frequency ID for frequency response, etc.)
            value      - load step value (e.g. eigenfrequency,
                         frequency, complex frequency, eigenvalue for
                         buckling, etc.)
        """
        self.point_data = point_data
        self.cell_data = cell_data
        self.id = step_id
        self.value = value


    def __repr__(self) -> str:
        lines = ["<meshio loadstep object>",
                 f"  Load step ID: {self.id:n}",
                 f"  Load value: {self.value:n}",
                 f"  Number of point data: {len(self.point_data)}",
                 f"  Number of cell data: {len(self.cell_data)}"]
        return "\n".join(lines)


    @property
    def id(self) -> int:
        return self._id


    @id.setter
    def id(self, step_id: int):
        self._id = int(step_id)


    @property
    def value(self) -> [float | complex]:
        return self._value


    @value.setter
    def value(self, value: [float | complex]):
        if type(value) is complex:
            self._value = value
        else:
            self._value = float(value)



class Data:
    def __init__(self, data: ArrayLike, ids: [list | ArrayLike], data_type: str=None,
                 data_characteristic: str=None, data_headers: list=None):
        """
        result data container
        In:
            data                - numpy array of same length as either points or cell,
                                  depending on the result data
            ids                 - point or cell ids of data lines
            data_type           - the type of data, default = unknown (unknown, force,
                                  temperature, heat flux, strain energy, displacement,
                                  reaction, kinetic energy, velocity, acceleration,
                                  strain energy density, kinetic energy density,
                                  pressure, heat, check, pressure coefficient)
            data_characteristic - type of values (unknown, scalar, vector, tensor)
            data_headers        - possible data column names, e.g.
                                  [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_xz]
                                  for stresses, if not specified then guess from
                                  data_type
        """
        self.type = data_type
        self.character = data_characteristic
        self.data = data
        self.headers = headers


    def __repr__(self) -> str:
        lines = ["<meshio data object>",
                 f"  Data type: {self.type:s}",
                 f"  Data character: {self.character:s}",
                 f"  Number of rows: {self.shape[0]:n}",
                 f"  Number of columns: {self.shape[1]:n}"]
        return "\n".join(lines)


    def __getitem__(self, id: int) -> ArrayLike:
        return self._data[id,:]


    def __iter__(self):
        return self._data.__iter__


    def __len__(self):
        return self._data.shape[0]


    def keys(self):
        return self.headers.__iter__


    def items(self):
        for i in range(self.columns.count):
            yield self.headers[i], self._data[:, i]


    @property
    def shape(self) -> tuple:
        return self._data.shape


    @property
    def columns_count(self) -> int:
        return self._data.shape[1]


    def columns(self) -> ArrayLike:
        for i in range(self._data.shape[1]):
            yield self._data[:,i]


    def point(self, pointid) -> ArrayLike:
        return self._data[pointid, :]


    def column(self, column: [int | str]) -> ArrayLike:
        if type(column) is str:
            return self._data[:, self.headers.index(column)]
        else:
            return self._data[:,column]


    @property
    def type(self) -> str:
        return self._type


    @type.setter
    def type(self, data_type: str=None):
        """
        In:
            data_type           - the type of data, default = unknown (unknown, force,
                                  temperature, heat flux, strain energy, displacement,
                                  reaction, kinetic energy, velocity, acceleration,
                                  strain energy density, kinetic energy density,
                                  pressure, heat, check, pressure coefficient)
        """
        if data_type is None:
            # data_type = "unknown"
            self._type = None
        elif (type(data_type) is not str) or (data_type not in meshio_data_types):
            self._type = "unknown"
        else:
            self._type = data_type


    @property
    def character(self) -> str:
        return self._character


    @character.setter
    def character(self, characteristic: str=None):
        """
        In:
            data_characteristic - type of values (unknown, scalar, vector, tensor)
        """
        if characteristic is None:
            self._character = None
        elif ((type(characteristic) is not str)
            or (characteristic not in meshio_data_characteristic)):
            self._character = "unknown"
        else:
            self._character = characteristic


    @property
    def headers(self) -> list:
        return self._headers



    @headers.setter
    def headers(self, headers: list=None):
        """
        In:
            data_headers        - possible data column names, e.g.
                                  [sig_xx, sig_yy, sig_zz, sig_xy, sig_yz, sig_xz]
                                  for stresses
        """
        self._headers = headers


    @property
    def data(self) -> ArrayLike:
        return self._data


    @data.setter
    def data(self, data: ArrayLike):
        """
        In:
            data                - numpy array of same length as either points or cell,
                                  depending on the result data
        """
        if type(data) is np.ndarray:
            self._data = data
            if type(data[0,0]) in (complex, np.csingle, np.cdouble):
                self._valtype = "complex"
            else:
                self._valtype = "real"
        else:
            self._data = np.array(data, dtype=float)
            self._valtype = "real"
        if len(self._data.shape) == 1:
            self._data = self._data.reshape(self.data.size, -1)

    @property
    def value_type(self) -> str:
        return self._valtype


class Mesh:
    def __init__(
        self,
        points: ArrayLike,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
        point_data: dict[str, ArrayLike] | None = None,
        cell_data: dict[str, list[ArrayLike]] | None = None,
        field_data=None,
        point_sets: dict[str, ArrayLike] | None = None,
        cell_sets: dict[str, list[ArrayLike]] | None = None,
        gmsh_periodic=None,
        info=None,
        point_gids: dict = None,   # original point ids
        # load_case: LoadCase = None,  # added load case container
    ):
        self.points = np.asarray(points)
        self.point_gids = point_gids   # point IDs
        if isinstance(cells, dict):
            # Let's not deprecate this for now.
            # warn(
            #     "cell dictionaries are deprecated, use list of tuples, e.g., "
            #     '[("triangle", [[0, 1, 2], ...])]',
            #     DeprecationWarning,
            # )
            # old dict, deprecated
            #
            # convert dict to list of tuples
            cells = list(cells.items())

        self.cells = []
        for cell_block in cells:
            if isinstance(cell_block, tuple):
                cell_type, data = cell_block
                cell_block = CellBlock(
                    cell_type,
                    # polyhedron data cannot be converted to numpy arrays
                    # because the sublists don't all have the same length
                    data if cell_type.startswith("polyhedron") else np.asarray(data),
                )
            self.cells.append(cell_block)

        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.field_data = {} if field_data is None else field_data
        self.point_sets = {} if point_sets is None else point_sets
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.gmsh_periodic = gmsh_periodic
        self.info = info

        # assert point data consistency and convert to numpy arrays
        # TODO:
        # only one load case possible
        # should have more levels e.g.:
        # point_data[load case] = {"id":     int id e.g. Mode Shape
        #                          "value":  float value e.g. Eigenfrequency
        #                          "info":   dict e.g. model type, analysis type..., voluntary
        #                          "vector": {"vector name": str vector_name e.g. "DISP",
        #                                     "type":        real or complex}
        #                                     "values":      np.array}
        #                         }
        # the same should apply to cell data and field data
        for key, item in self.point_data.items():
            if not type(item) is dict: # keep legacy, add new
                self.point_data[key] = np.asarray(item)
                if len(self.point_data[key]) != len(self.points):
                    raise ValueError(
                        f"len(points) = {len(self.points)}, "
                        f'but len(point_data["{key}"]) = {len(self.point_data[key])}'
                    )

        # assert cell data consistency and convert to numpy arrays
        for key, data in self.cell_data.items():
            if len(data) != len(cells):
                raise ValueError(
                    f"Incompatible cell data '{key}'. "
                    f"{len(cells)} cell blocks, but '{key}' has {len(data)} blocks."
                )

            for k in range(len(data)):
                data[k] = np.asarray(data[k])
                if len(data[k]) != len(self.cells[k]):
                    raise ValueError(
                        "Incompatible cell data. "
                        + f"Cell block {k} ('{self.cells[k].type}') "
                        + f"has length {len(self.cells[k])}, but "
                        + f"corresponding cell data item has length {len(data[k])}."
                    )

    def __repr__(self):
        lines = ["<meshio mesh object>", f"  Number of points: {len(self.points)}"]
        special_cells = [
            "polygon",
            "polyhedron",
            "VTK_LAGRANGE_CURVE",
            "VTK_LAGRANGE_TRIANGLE",
            "VTK_LAGRANGE_QUADRILATERAL",
            "VTK_LAGRANGE_TETRAHEDRON",
            "VTK_LAGRANGE_HEXAHEDRON",
            "VTK_LAGRANGE_WEDGE",
            "VTK_LAGRANGE_PYRAMID",
        ]
        if len(self.cells) > 0:
            lines.append("  Number of cells:")
            for cell_block in self.cells:
                string = cell_block.type
                if cell_block.type in special_cells:
                    string += f"({cell_block.data.shape[1]})"
                lines.append(f"    {string}: {len(cell_block)}")
        else:
            lines.append("  No cells.")

        if self.point_sets:
            names = ", ".join(self.point_sets.keys())
            lines.append(f"  Point sets: {names}")

        if self.cell_sets:
            names = ", ".join(self.cell_sets.keys())
            lines.append(f"  Cell sets: {names}")

        if self.point_data:
            # print(f"{self.point_data = }")
            names = ", ".join([str(k) for k in self.point_data.keys()])
            lines.append(f"  Point data: {names}")

        if self.cell_data:
            names = ", ".join(self.cell_data.keys())
            lines.append(f"  Cell data: {names}")

        if self.field_data:
            names = ", ".join(self.field_data.keys())
            lines.append(f"  Field data: {names}")

        # if self.load_case:
        #     names = ", ".join([str(lc.id) for lc in self.load_case])
        #     lines.append(f"  Load cases: {names}")

        return "\n".join(lines)

    def copy(self):
        return copy.deepcopy(self)

    def write(self, path_or_buf, file_format: str | None = None, **kwargs):
        # avoid circular import
        from ._helpers import write

        write(path_or_buf, self, file_format, **kwargs)

    def get_cells_type(self, cell_type: str):
        if not any(c.type == cell_type for c in self.cells):
            return np.empty((0, num_nodes_per_cell[cell_type]), dtype=int)
        return np.concatenate([c.data for c in self.cells if c.type == cell_type])

    def get_cell_data(self, name: str, cell_type: str):
        return np.concatenate(
            [d for c, d in zip(self.cells, self.cell_data[name]) if c.type == cell_type]
        )

    @property
    def cells_dict(self):
        cells_dict = {}
        for cell_block in self.cells:
            if cell_block.type not in cells_dict:
                cells_dict[cell_block.type] = []
            cells_dict[cell_block.type].append(cell_block.data)
        # concatenate
        for key, value in cells_dict.items():
            cells_dict[key] = np.concatenate(value)
        return cells_dict

    @property
    def cells_dict_paraview(self):
        cells_dict = {}
        for cell_block in self.cells:
            if cell_block.type not in cells_dict:
                cells_dict[cell_block.type] = []
            cells_dict[cell_block.type].append(cell_block.data)
        # concatenate
        # for key, value in cells_dict.items():
        #     cells_dict[key] = np.concatenate(value)
        return cells_dict

    @property
    def cell_data_dict(self):
        cell_data_dict = {}
        for key, value_list in self.cell_data.items():
            cell_data_dict[key] = {}
            for value, cell_block in zip(value_list, self.cells):
                if cell_block.type not in cell_data_dict[key]:
                    cell_data_dict[key][cell_block.type] = []
                cell_data_dict[key][cell_block.type].append(value)

            for cell_type, val in cell_data_dict[key].items():
                cell_data_dict[key][cell_type] = np.concatenate(val)
        return cell_data_dict

    @property
    def cell_sets_dict(self):
        sets_dict = {}
        for key, member_list in self.cell_sets.items():
            sets_dict[key] = {}
            offsets = {}
            for members, cells in zip(member_list, self.cells):
                if members is None:
                    continue
                if cells.type in offsets:
                    offset = offsets[cells.type]
                    offsets[cells.type] += cells.data.shape[0]
                else:
                    offset = 0
                    offsets[cells.type] = cells.data.shape[0]
                if cells.type in sets_dict[key]:
                    sets_dict[key][cells.type].append(members + offset)
                else:
                    sets_dict[key][cells.type] = [members + offset]
        return {
            key: {
                cell_type: np.concatenate(members)
                for cell_type, members in sets.items()
                if sum(map(np.size, members))
            }
            for key, sets in sets_dict.items()
        }

    @classmethod
    def read(cls, path_or_buf, file_format=None):
        # avoid circular import
        from ._helpers import read

        # 2021-02-21
        warn("meshio.Mesh.read is deprecated, use meshio.read instead")
        return read(path_or_buf, file_format)

    def cell_sets_to_data(self, data_name: str | None = None):
        # If possible, convert cell sets to integer cell data. This is possible if all
        # cells appear exactly in one group.
        default_value = -1
        if len(self.cell_sets) > 0:
            intfun = []
            for k, c in enumerate(zip(*self.cell_sets.values())):
                # Go for -1 as the default value. (NaN is not int.)
                arr = np.full(len(self.cells[k]), default_value, dtype=int)
                for i, cc in enumerate(c):
                    if cc is None:
                        continue
                    arr[cc] = i
                intfun.append(arr)

            for item in intfun:
                num_default = np.sum(item == default_value)
                if num_default > 0:
                    warn(
                        f"{num_default} cells are not part of any cell set. "
                        f"Using default value {default_value}."
                    )
                    break

            if data_name is None:
                data_name = "-".join(self.cell_sets.keys())
            self.cell_data[data_name] = intfun
            self.cell_sets = {}

    def point_sets_to_data(self, join_char: str = "-") -> None:
        # now for the point sets
        # Go for -1 as the default value. (NaN is not int.)
        default_value = -1
        if len(self.point_sets) > 0:
            intfun = np.full(len(self.points), default_value, dtype=int)
            for i, cc in enumerate(self.point_sets.values()):
                intfun[cc] = i

            if np.any(intfun == default_value):
                warn(
                    "Not all points are part of a point set. "
                    f"Using default value {default_value}."
                )

            data_name = join_char.join(self.point_sets.keys())
            self.point_data[data_name] = intfun
            self.point_sets = {}

    # This used to be int_data_to_sets(), converting _all_ cell and point data.
    # This is not useful in many cases, as one usually only wants one
    # particular data array (e.g., "MaterialIDs") converted to sets.
    def cell_data_to_sets(self, key: str):
        """Convert point_data to cell_sets."""
        data = self.cell_data[key]

        # handle all int and uint data
        if not all(v.dtype.kind in ["i", "u"] for v in data):
            raise RuntimeError(f"cell_data['{key}'] is not int data.")

        tags = np.unique(np.concatenate(data))

        # try and get the names by splitting the key along "-" (this is how
        # sets_to_int_data() forms the key)
        names = key.split("-")
        # remove duplicates and preserve order
        # <https://stackoverflow.com/a/7961390/353337>:
        names = list(dict.fromkeys(names))
        if len(names) != len(tags):
            # alternative names
            names = [f"set-{key}-{tag}" for tag in tags]

        # TODO there's probably a better way besides np.where, something from
        # np.unique or np.sort
        for name, tag in zip(names, tags):
            self.cell_sets[name] = [np.where(d == tag)[0] for d in data]

        # remove the cell data
        del self.cell_data[key]

    def point_data_to_sets(self, key: str):
        """Convert point_data to point_sets."""
        data = self.point_data[key]

        # handle all int and uint data
        if not all(v.dtype.kind in ["i", "u"] for v in data):
            raise RuntimeError(f"point_data['{key}'] is not int data.")

        tags = np.unique(data)

        # try and get the names by splitting the key along "-" (this is how
        # sets_to_int_data() forms the key
        names = key.split("-")
        # remove duplicates and preserve order
        # <https://stackoverflow.com/a/7961390/353337>:
        names = list(dict.fromkeys(names))
        if len(names) != len(tags):
            # alternative names
            names = [f"set-key-{tag}" for tag in tags]

        # TODO there's probably a better way besides np.where, something from
        # np.unique or np.sort
        for name, tag in zip(names, tags):
            self.point_sets[name] = np.where(data == tag)[0]

        # remove the cell data
        del self.point_data[key]

