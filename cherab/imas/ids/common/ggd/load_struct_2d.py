# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

import numpy as np

from imas.imasdef import EMPTY_INT

from cherab.imas.ggd import StructGrid2D


NODE_DIMENSION = 0
FACE_DIMENSION = 2


def load_struct_grid_2d(grid_ggd, space_indices=(0, 1), with_subsets=False):
    """
    Loads structured 2D grid from the grid_ggd structure.

    :param grid_ggd: The grid_ggd structure.
    :param space_indices: The 2-tuple of spaces indices to use. Default is (0, 1).
    :param with_subsets: Read grid subset data if True. Default is True.

    :returns:
    |   grid: An StructGrid2D instance.
    |   subsets (optional): Dictionary with grid subsets for each subset name containing the indices
    |       of the cells from that subset. Note that 'Cells' subset is included only if cell indices
    |       are specified.
    |   subset_id (optional): Dictionary with grid subset identifier indices.
    """

    if len(space_indices > 2):
        raise ValueError("The argument 'space_indices' must contain only two indices.")

    spaces = [grid_ggd.space[i] for i in space_indices]
    
    # Check if the grid is structured
    if len(spaces[0].objects_per_dimension) >= 3 or len(spaces[1].objects_per_dimension) >= 3:
        raise ValueError("The load_struct_grid_2d() supports only structured 2D grids.")

    grid_name = grid_ggd.identifier.name

    # Reading grid nodes
    nodes = []
    for space in spaces:
        space_nodes = [object.geometry[0] for object in space.objects_per_dimension[NODE_DIMENSION].object]
        nodes.append(space_nodes)
    
    grid = StructGrid2D(nodes[0], nodes[1], name=grid_name)
    
    if not with_subsets:
        return grid
    
    # Reading grid subsets (2D only)
    cell_subset_ids = (5, 22, 23, 24, 25, 38, 39, 40)

    subsets = {}
    subset_id = {}
    for subset in grid_ggd.grid_subset:
        dimension_is_2d = (subset.dimension == FACE_DIMENSION + 1)  # C to Fortran indexing
        known_subset_id = (subset.dimension == EMPTY_INT and subset.identifier.index in cell_subset_ids)
        if (dimension_is_2d or known_subset_id) and len(subset.element):
            name = subset.identifier.name
            # cell indices
            indices = (len(nodes[0]) * np.ones(len(subset.element), dtype=np.int32),
                       len(nodes[1]) * np.ones(len(subset.element), dtype=np.int32))
            # the element represents the cell
            for i, element in enumerate(subset.element):
                # we need only the bottom left corner indices ot label the cell
                for object in element.object:
                    if object.dimension > NODE_DIMENSION:
                        # not a node
                        continue
                    space_index = object.space - 1  # FORTRAN to C indexing
                    if space_index not in space_indices:
                        # wrong space
                        continue
                    ispace = space_indices.index(space_index)
                    indices[ispace][i] = min(indices[ispace][i], object.index - 1)  # FORTRAN to C indexing
            subsets[name] = indices
            subset_id[name] = subset.identifier.index

    return grid, subsets, subset_id
