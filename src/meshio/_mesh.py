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


class ID:
    def __init__(self, id: int):
        self.id = id

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, id: int):
        self._id = int(id)


class Analysis:
    def __init__(self, point_data: dict, value_count: int):
        self._loadcases = []
        self._loadcase_ids = []
        self.value_count = value_count
        self.loadcases = point_data
        self.analysis = point_data

    def __getitem__(self, load_case_id: int):
        return self.loadcases[self.loadcaseIDs.index(load_case_id)]

    def __iter__(self):
        return self.loadcases.__iter__

    def keys(self) -> int:
        for i in range(len(self._loadcase_ids)):
            yield self._loadcase_ids[i]

    def items(self) -> tuple:
        for i in range(len(self._loadcase_ids)):
            yield (self._loadcase_ids[i], self._loadcases[i])

    def values(self) -> LoadCase:
        for i in range(len(self._loadcase_ids)):
            yield self._loadcases[i]

    def as_dict(self) -> dict:
        data = {}
        for lcid, lcase in self.items():
            name = self.analysis.replace(" ", "_") + "-" + f"{lcid:05n}"
            for lsid, lstep in lcase.items():
                name += f"-{lsid:05n}({lstep.value:f})"
                for dtname, dataset in lstep.items():
                    name += f"-{dtname:s}"
                    data[name] = dataset.data.as_array()
        return data

    def as_iter(self) -> tuple[str, ArrayLike]:
        for lcid, lcase in self.items():
            for lsid, lstep in lcase.items():
                for dtname, dataset in lstep.items():
                    yield (
                        self.analysis, lcid, lsid, dataset.character,
                        dataset.type, dataset.as_array()
                    )

    @property
    def analysis(self) -> str:
        return self._analysis

    @analysis.setter
    def analysis(self, analysis: [str, dict]):
        if type(analysis) is dict and "analysis" in analysis.keys():
            analysis = analysis["analysis"]
        if analysis in meshio_analysis:
            self._analysis = analysis
        else:
            warn(f"Unknown type of analysis {analysis:s}, setting it to 'unknown'.")
            self._analysis = "unknown"

    @property
    def loadcases(self) -> list:
        return self._loadcases

    @property
    def loadcaseIDs(self) -> list:
        return self._loadcase_ids

    @loadcases.setter
    def loadcases(self, point_data: dict):
        """
        !!! Only for POINT DATA !!!

        -> point_data[load case][load step][data type][data] = [..., ..., ...]
        -> point_data: analysis: analysis type - str
                       load case 1: load case ID - (int, dict)
                       load case 2: load case ID - (int, dict)

        -> load case 1: load step 1 - (int, dict)
                        load step 2 - (int, dict)

        -> load step 1: value       - float (e.g. time, eigenfrequency etc.)
                        data type 1 - (str, dict) - (e.g. stress, strain, force)
                        data type 2 - (str, dict)

        -> data type 1: data character - str - (e.g scalar, vector3, tensor)
                        value type     - str - (e.g. integer, sp real, dp complex, ...)
                        gids           - list - original Node IDs
                        data           - np.ndarray(value type) - values

        """
        lcases = [k for k in point_data.keys() if k != "analysis"]
        for lcase in lcases:
            self._loadcase_ids.append(lcase)
            self._loadcases.append(LoadCase(lcase, point_data[lcase], self.value_count))

        # sort by loadcase id
        self._loadcase_ids = sorted(self._loadcase_ids)
        self._loadcases = sorted(self._loadcases, key=lambda x: x.id)


class LoadCase(ID):
    def __init__(self, load_case_id: int, load_steps: dict, value_count: int):
        super().__init__(load_case_id)
        self._loadsteps = []
        self._loadstep_ids = []
        self.loadsteps = load_steps

    @property
    def loadsteps(self) -> list:
        return self.loadsteps

    @loadsteps.setter
    def loadsteps(self, loadsteps: dict):
        for ls in loadsteps.keys():
            self._loadstep_ids.append(ls)
            lss = {dt: loadstep for dt, loadstep in loadsteps[ls].items() if dt != "value"}
            self._loadsteps.append(LoadStep(ls, lss, self.value_count, loadsteps[ls]["value"]))

        # sort by loadstep.id, just in case
        self._loadstep_ids = sorted(self._loadstep_ids)
        self._loadsteps = sorted(self._loadsteps, key=lambda x: x.id)

    @property
    def loadstepIDs(self) -> list:
        return self._loadstep_ids

    def __getitem__(self, id: [int, float]) -> LoadStep:
        if type(id) is int:
            return self.loadsteps[self.loadstepIDs.index(id)]
        else:
            for ls in self.loadsteps:
                if ls.value == id:
                    return ls

    def __iter__(self) -> LoadStep:
        for ls in self.loadsteps:
            yield ls

    def keys(self) -> list:
        return self.loadstepIDs

    def items(self) -> tuple:
        for i in range(len(self.loadsteps)):
            yield self._loadstep_ids[i], self._loadsteps[i]

    def values(self) -> LoadStep:
        for i in range(len(self.loadsteps)):
            yield self._loadsteps[i]


class LoadStep(ID):
    def __init__(self, load_step_id: int, results: dict, value_count: int,
                 load_step_value: [float, complex] = 0.0):
        super().__init__(load_step_id)
        self._value_count = value_count
        self.value = load_step_value
        self._results = []
        self._result_types = []
        self.results = results

    def __getitem__(self, result_type) -> Data:
        return self.results[self.result_types.index(result_type)]

    def __repr__(self) -> str:
        lines = ["<meshio LoadStep object>",
                 f"  Load Step ID: {self.id:n}",
                 f"  Load Step Value: {self.value:n}",
                 f"  Number of results: {len(self.results):n}"]
        for result in self.results:
            lines.append(("  " + result.replace("\n", "\n  ")).split("\n"))
        return "\n".join(lines)

    def keys(self) -> list:
        return self.result_types.__iter__

    def items(self) -> tuple:
        for i in range(len(self.results)):
            yield (self.result_types[i], self.results[i])

    def values(self) -> list:
        for i in range(len(self.results)):
            yield self.results[i]

    @property
    def value(self) -> [float | complex]:
        return self._value

    @value.setter
    def value(self, load_step_value: [float, complex]):
        self._value = load_step_value

    @property
    def results(self) -> list:
        return self._results

    @results.setter
    def results(self, results: dict):
        for key, item in results.items():
            self._results.append(PointData(key, item["data"], self._value_count,
                                           item["gids"]))
            self._result_types.append(key)
            if "character" in item.keys():
                self._results[-1].character = item["character"]

    @property
    def result_types(self) -> list:
        return self._result_types


class PointData:
    def __init__(self, data_type: str, point_data: [ArrayLike], point_count: int,
                 gids: list=None, character: str = "unknown"):
        self._count = point_count
        self.type = data_type
        self.gids = gids
        self.data = point_data
        self.character = character

    def __getitem__(self, point_id: int) -> ArrayLike:
        """
        Item getter that returns value if point_id is in gids, otherwise
        returns a vector of zeros the same size as other data

        This way zeroes need not be stored in memory.
        """
        if self.gids is not None:
            if point_id in self.gids:
                return self._data[self.gids.index(point_id),:]
            else:
                return np.array([0.] * self._data.shape[1], dtype=self._data.dtype)
        else:
            return self._data[point_id]

    def __len__(self):
        return self._count

    def __repr__(self) -> str:
        lines = ["<meshio Data object>",
                 f"  Data type: {self.type:s}",
                 f"  Data character: {self.character:s}",
                 f"  Data shape: {self.shape[0]:n}, {self.shape[1]:n}"]
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return self._count

    def keys(self):
        return self.gids

    def items(self):
        if self.gids is not None:
            for i in range(len(self.gids)):
                yield (self.gids[i], self.data[i, :])
        else:
            for i in range(len(self.data)):
                yield (i, self.data[i, :])

    def values(self):
        return self.data.__iter__

    def as_array(self, num_points: int=None):
        if self.gids is None:
            return self.data
        else:
            data = []
            vals = self._data.shape[1]
            for i in range(self.count):
                if i in self.gids.keys():
                    data.append(self._data[i,:])
                else:
                    data.append([0.] * vals)
            data = np.array(data, dtype = self._data.dtype)
            return data

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, data_type: str):
        self._type = data_type

    @property
    def gids(self) -> list:
        return self._gids

    @gids.setter
    def gids(self, gids: list):
        if gids is None:
            self._gids = None
        else:
            self._gids = list(gids)

    @property
    def data(self) -> ArrayLike:
        return self._data

    @data.setter
    def data(self, point_data: ArrayLike):
        self._data = np.asarray(point_data)
        if self.gids is None:
            self._count = self._data.shape[0]

    @property
    def character(self) -> str:
        return self._character

    @character.setter
    def character(self, character: str):
        if character not in meshio_data_characteristic.keys():
            self._character = "unknown"
        else:
            self._character = character

        if self._character != "unknown":
            colcnt = len(meshio_data_characteristic[self._character])
            if self._data.shape[1] != colcnt:
                warn(f"Point data character {self._character:s} does not match "
                     f"point data columns count ({colcnt:n} != {self._data.shape[1]:n}), "
                      "setting data character to 'unknown'.")
                self._character = "unknown"

    @property
    def columns(self) -> list:
        cols = meshio_data_characteristic[self.character]
        if self.character == "unknown":
            return [cols[0] + "f{i+1:n}" for i in range(self._data.shape[1])]
        else:
            return cols


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

        # self.point_data = {} if point_data is None else point_data
        self.point_data = point_data
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
        # for key, item in self.point_data.items():
        #     if not type(item) is dict: # keep legacy, add new
        #         self.point_data[key] = np.asarray(item)
        #         if len(self.point_data[key]) != len(self.points):
        #             raise ValueError(
        #                 f"len(points) = {len(self.points)}, "
        #                 f'but len(point_data["{key}"]) = {len(self.point_data[key])}'
        #             )

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
    def cells_dict_paraview(self) -> dict:
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

    @property
    def point_data(self) -> dict:
        return self._point_data

    @point_data.setter
    def point_data(self, point_data: dict):
        """
        -> analysis[load case][load step][data type][data] = [..., ..., ...]
        -> analysis: analysis: analysis type - str
                     load case 1: load case ID - (int, dict)
                     load case 2: load case ID - (int, dict)

        -> load case 1: load step 1 - (int, dict)
                        load step 2 - (int, dict)

        -> load step 1: value       - float (e.g. time, eigenfrequency etc.)
                        data type 1 - (str, dict) - (e.g. stress, strain, force)
                        data type 2 - (str, dict)

        -> data type 1: data character - str - (e.g scalar, vector3, tensor)
                        value type     - str - (e.g. integer, sp real, dp complex, ...)
                        gids           - list - original Node IDs
                        data           - np.ndarray(value type) - values

        """
        if point_data is None:
            point_data = {}
        elif type(point_data) is Analysis:
            pass
        else:
            for key, item in point_data.items():
                if not type(item) is dict: # keep legacy, add new
                    point_data[key] = np.asarray(item)
                    if len(point_data[key]) != len(self.points):
                        raise ValueError(
                            f"len(points) = {len(self.points)}, "
                            f'but len(point_data["{key}"]) = {len(point_data[key])}'
                        )
                else:
                    point_data = Analysis(point_data)
                    break
        self._point_data = point_data

    @property
    def point_data_load_cases(self) -> list:
        return list(self.point_data.keys())

    def point_data_load_steps(self, load_case: int) -> dict:
        return list([k for k in self.point_data[load_case].keys() if k != "analysis"])


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

