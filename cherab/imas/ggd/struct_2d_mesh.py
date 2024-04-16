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
import matplotlib.pyplot as plt

from raysect.core.math import Vector3D

from cherab.imas.math import StructGridFunction2D, StructGridVectorFunction2D
from .base_mesh import GGDGrid


class StructGrid2D(GGDGrid):
    """
    Structured 2D grid object.

    The grid cells are rectangles, defined by x and y nodes.

    :param object x: Array-like of shape (N,) containing coordinates of the nodes along 1st axis.
    :param object y: Array-like of shape (M,) containing coordinates of the nodes along 2nd axis.
    :param object subset_indices: A (2, K)-shaped tuple containing cell indices
        of the grid subset. Default is None (the subset contains the whole grid).
    :param str name: A name of the grid. Default is 'Cells'.
    :param str coordinate_system: The coordinate system: 'cylindrical' (default)
        or 'cartesian'. In case of 'cylindrical' coordinates, x and y define the rz plane.
    
    :ivar name: A name of the grid.
    :ivar dimension: Grid dimensions.
    :ivar num_cell: The number of grid cells in the subset.
    :ivar x: The coordinates of the nodes along 1st axis.
    :ivar y: The coordinates of the nodes along 2nd axis.
    :ivar subset_indices: A (2, num_cell)-shaped tuple containing cell indices
        of the grid subset.
    :ivar cell_centre: Coordinates of cell centres as (num_cell,2) array.
    :ivar cell_area: Cell areas as (num_cell,) array.
    :ivar cell_volume: Cell volumes as (num_cell,) array in case of cylindrical coordinates.
    :ivar mesh_extent: Extent of the mesh. A dictionary with xmin, xmax, ymin, ymax keys.
        In case of cylindrical coordinates also contains rmin, rmax, zmin, zmax keys.
    """

    def __init__(self, x, y, subset_indices=None, name='Cells', coordinate_system='cylindrical'):

        x = np.sort(np.array(x, dtype=np.float64))
        x.setflags(write=False)
        y = np.sort(np.array(y, dtype=np.float64))
        y.setflags(write=False)

        if x.ndim != 1:
            raise ValueError("Attribute 'x' must be a 1D array-like. The number of dimensions in 'x' is {}.".format(x.ndim))

        if y.ndim != 1:
            raise ValueError("Attribute 'y' must be a 1D array-like. The number of dimensions in 'y' is {}.".format(y.ndim))

        if x.size < 2:
            raise ValueError("Attribute 'x' must contain at least two elements.")

        if y.size < 2:
            raise ValueError("Attribute 'y' must contain at least two elements.")

        self._x = x
        self._y = y
        self._set_subset_indices(subset_indices)

        super().__init__(name, 2, coordinate_system)

    def _initial_setup(self):

        self._interpolator = None

        self._num_cell = self._subset_indices[0].size

        imin = self._subset_indices[0].min()
        jmin = self._subset_indices[1].min()
        imax = self._subset_indices[0].max() + 1
        jmax = self._subset_indices[1].max() + 1

        if self._coordinate_system == 'cylindrical':
            self._mesh_extent = {
                "xmin": -self._x[imax], "xmax": self._x[imax],
                "ymin": -self._x[imax], "ymax": self._x[imax],
                "rmin": self._x[imin], "rmax": self._x[imax],
                "zmin": self._y[jmin], "zmax": self._y[jmax]
            }
        elif self._coordinate_system == 'cartesian':
            self._mesh_extent = {
                "xmin": self._x[imin], "xmax": self._x[imax],
                "ymin": self._y[jmin], "ymax": self._y[jmax]
            }

        # Calculate cell area and centroid
        xc = 0.5 * (self._x[1:] + self._x[:-1])
        yc = 0.5 * (self._y[1:] + self._y[:-1])
        self._cell_centre = np.array([xc[self._subset_indices[0]],
                                      yc[self._subset_indices[1]]]).T

        dx = np.diff(self._x)[self._subset_indices[0]]
        dy = np.diff(self._y)[self._subset_indices[1]]
        self._cell_area = dx * dy

        self._cell_centre.setflags(write=False)
        self._cell_area.setflags(write=False)

        if self._coordinate_system == 'cylindrical':
            self._cell_volume = 2 * np.pi * self._cell_centre[:, 0] * self._cell_area
            self._cell_volume.setflags(write=False)
    
    @property
    def x(self):
        """Mesh x nodes."""
        return self._x

    @property
    def y(self):
        """Mesh y nodes."""
        return self._y
    
    @property
    def subset_indices(self):
        """Grid subset indices."""
        return self._subset_indices
    
    @subset_indices.setter
    def subset_indices(self, value):
        self._set_subset_indices(value)
        self._initial_setup()

    def _set_subset_indices(self, value):
        if value is not None:
            value = np.array(value, dtype=np.int32)

            if value.ndim != 2:
                raise ValueError("Argument 'subset_indices' must be 2D array.")

            if value.shape[0] != 2:
                raise ValueError("Argument 'subset_indices' must be a (2, N)-shaped array.")

            ind1d = value[0] * self._y.size + value[1]
            _, counts = np.unique(ind1d, return_counts=True)
            if np.any(counts > 1):
                raise ValueError("Argument 'subset_indices' must contain only unique indices.")
            self._subset_indices = (value[0], value[1])
        else:
            i, j = np.meshgrid(np.arange(self._x.size - 1, dtype=np.int32),
                               np.arange(self._y.size - 1, dtype=np.int32),
                               indexing='ij')
            self._subset_indices = (i.flatten(), j.flatten())

        self._subset_indices[0].setflags(write=False)
        self._subset_indices[1].setflags(write=False)
    
    def subset(self, indices, name=None):
        """
        Creates a subset StructGrid2D from this instance.
    
        If indices is the 2D (2, K)-shaped tuple, the subset is created with respect
        to the entire grid.
        If indices is the 1D array, the subset is created with respect
        to the current subset.

        :param indices: Indices of the cells with respect to the entire grid
                        or to the current subset.
        :param name: Name of the grid subset. Default is instance.name + ' subset'.
        """

        grid = StructGrid2D.__new__(StructGrid2D)

        grid._name = name or self.name + ' subset'
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._x = self._x
        grid._y = self._y

        indices = np.array(indices, dtype=np.int32)

        if indices.ndim == 1:
            # Create a new subset from the current subset
            _, counts = np.unique(indices, return_counts=True)
            if np.any(counts > 1):
                raise ValueError("Argument 'indices' must contain only unique indices.")

            if indices.max() >= self._num_cell:
                raise ValueError("The new subset is larger than the current subset.")
            
            grid._subset_indices = (self._subset_indices[0][indices], self._subset_indices[1][indices])
            grid._initial_setup()

        elif indices.ndim == 2:
            # Create a subset from the entire grid
            grid.subset_indices = (indices[0], indices[1])

        else:
            raise ValueError("Argument 'indices' must be 1D or 2D array.")

        return grid
    
    def interpolator(self, grid_data, fill_value=0):
        """
        Returns an StructGridFunction2D interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_data: An array containing data in the grid cells.
        :param fill_value: A value returned outside the gird. Default is 0.

        :returns: StructGridFunction2D interpolator
        """
        if self._interpolator is None:
            self._interpolator = StructGridFunction2D(self._x, self._y, self._subset_indices, grid_data, fill_value)
            return self._interpolator
        
        return StructGridFunction2D.instance(self._interpolator, grid_data, fill_value)

    def vector_interpolator(self, grid_vectors, fill_vector=Vector3D(0, 0, 0)):
        """
        Returns an StructGridVectorFunction2D interpolator instance for the vector data
        defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_vectors: A (3,K) array containing 3D vectors in the grid cells.
        :param fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).

        :returns: StructGridVectorFunction2D interpolator
        """
        if self._interpolator is None:
            self._interpolator = StructGridVectorFunction2D(self._x, self._y, self._subset_indices, grid_vectors, fill_vector)
            return self._interpolator

        return StructGridVectorFunction2D.instance(self._interpolator, grid_vectors, fill_vector)

    def __getstate__(self):
        state = {
            'name': self._name,
            'dimension': self._dimension,
            'coordinate_system': self._coordinate_system,
            'x': self._x,
            'y': self._y,
            'subset_indices': self._subset_indices
        }
        return state

    def __setstate__(self, state):
        self._name = state['name']
        self._dimension = state['dimension']
        self._coordinate_system = state['coordinate_system']
        self._x = state['x']
        self._x.setflags(write=False)
        self._y = state['y']
        self._y.setflags(write=False)
        self._subset_indices = tuple(state['subset_indices'])
        self._subset_indices[0].setflags(write=False)
        self._subset_indices[1].setflags(write=False)

        self._initial_setup()

    def plot_mesh(self, data=None, ax=None):
        """
        Plot the geometry of the grid subset to a matplotlib figure.

        :param data: Data array defined on the subset.
        """

        if ax is None:
            _, ax = plt.subplots(constrained_layout=True)
        
        data2plot = np.zeros((self._x.size - 1, self._y.size - 1))
        data2plot[:, :] = np.nan

        if data is None:
            data = np.ones(self._num_cell)
        
        data2plot[self._subset_indices] = data

        ax.pcolormesh(self._x, self._y, data2plot.T)
        ax.set_aspect(1)

        if self._coordinate_system == 'cartesian':
            ax.set_xlim(self._mesh_extent["xmin"], self._mesh_extent["xmax"])
            ax.set_ylim(self._mesh_extent["ymin"], self._mesh_extent["ymax"])
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
        elif self._coordinate_system == 'cylindrical':
            ax.set_xlim(self._mesh_extent["rmin"], self._mesh_extent["rmax"])
            ax.set_ylim(self._mesh_extent["zmin"], self._mesh_extent["zmax"])
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")            

        return ax
