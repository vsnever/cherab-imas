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

from raysect.core.math import Vector3D
from cherab.core.math.transform import CylindricalTransform, VectorCylindricalTransform

from cherab.imas.math import StructGridFunction3D, StructGridVectorFunction3D
from .base_mesh import GGDGrid


class StructGrid3D(GGDGrid):
    """
    Structured 3D grid object.

    The grid cells are rectangular cuboids, defined by x, y and z nodes.

    :param object x: Array-like of shape (N,) containing coordinates of the nodes along 1st axis.
    :param object y: Array-like of shape (M,) containing coordinates of the nodes along 2nd axis.
    :param object z: Array-like of shape (L,) containing coordinates of the nodes along 3rd axis.
    :param object subset_indices: A (3, K)-shaped tuple containing cell indices
        of the grid subset. Default is None (the subset contains the whole grid).
    :param str name: A name of the grid. Default is 'Cells'.
    :param str coordinate_system: The coordinate system: 'cylindrical' or 'cartesian' (default).
        In case of 'cylindrical' coordinates, x and z define the rz plane and y
        is the polar angle in radians.

    :ivar name: A name of the grid.
    :ivar dimension: Grid dimensions.
    :ivar num_cell: The number of grid cells in the subset.
    :ivar x: The coordinates of the nodes along 1st axis.
    :ivar y: The coordinates of the nodes along 2nd axis.
    :ivar z: The coordinates of the nodes along 3rd axis.
    :ivar subset_indices: A (3, num_cell)-shaped tuple containing cell indices
        of the grid subset.
    :ivar cell_centre: Coordinates of cell centres as (num_cell,3) array.
    :ivar cell_volume: Cell volumes as (num_cell,) array.
    :ivar mesh_extent: Extent of the mesh. A dictionary with:
        xmin, xmax, ymin, ymax, zmin, zmax, rmin, rmax, phimin, phimax keys.
    """

    def __init__(self, x, y, z, subset_indices=None, name='Cells', coordinate_system='cartesian'):

        x = np.sort(np.array(x, dtype=np.float64))
        x.setflags(write=False)
        y = np.sort(np.array(y, dtype=np.float64))
        y.setflags(write=False)
        z = np.sort(np.array(z, dtype=np.float64))
        z.setflags(write=False)

        if x.ndim != 1:
            raise ValueError("Attribute 'x' must be a 1D array-like. The number of dimensions in 'x' is {}.".format(x.ndim))

        if y.ndim != 1:
            raise ValueError("Attribute 'y' must be a 1D array-like. The number of dimensions in 'y' is {}.".format(y.ndim))

        if z.ndim != 1:
            raise ValueError("Attribute 'z' must be a 1D array-like. The number of dimensions in 'z' is {}.".format(z.ndim))

        if x.size < 2:
            raise ValueError("Attribute 'x' must contain at least two elements.")

        if y.size < 2:
            raise ValueError("Attribute 'y' must contain at least two elements.")

        if z.size < 2:
            raise ValueError("Attribute 'z' must contain at least two elements.")

        self._x = x
        self._y = y
        self._z = z
        self._set_subset_indices(subset_indices)

        super().__init__(name, 3, coordinate_system)

    def _initial_setup(self):

        self._interpolator = None

        self._num_cell = self._subset_indices[0].size

        imin = self._subset_indices[0].min()
        jmin = self._subset_indices[1].min()
        kmin = self._subset_indices[2].min()
        imax = self._subset_indices[0].max() + 1
        jmax = self._subset_indices[1].max() + 1
        kmax = self._subset_indices[2].max() + 1

        if self._coordinate_system == 'cylindrical':
            x00 = self._x[imin] * np.cos(self._y[jmin])
            x01 = self._x[imin] * np.cos(self._y[jmax])
            x10 = self._x[imax] * np.cos(self._y[jmin])
            x11 = self._x[imax] * np.cos(self._y[jmax])
            xmin = min(x00, x01, x10, x11)
            xmax = max(x00, x01, x10, x11)

            y00 = self._x[imin] * np.sin(self._y[jmin])
            y01 = self._x[imin] * np.sin(self._y[jmax])
            y10 = self._x[imax] * np.sin(self._y[jmin])
            y11 = self._x[imax] * np.sin(self._y[jmax])
            ymin = min(y00, y01, y10, y11)
            ymax = max(y00, y01, y10, y11)

            self._mesh_extent = {
                "xmin": xmin, "xmax": xmax,
                "ymin": ymin, "ymax": ymax,
                "rmin": self._x[imin], "rmax": self._x[imax],
                "phimin": self._y[jmin], "phimax": self._y[jmax],
                "zmin": self._z[kmin], "zmax": self._z[kmax]
            }
        elif self._coordinate_system == 'cartesian':
            r00 = self._x[self._subset_indices[0]]**2 + self._y[self._subset_indices[1]]**2
            r01 = self._x[self._subset_indices[0]]**2 + self._y[self._subset_indices[1] + 1]**2
            r10 = self._x[self._subset_indices[0] + 1]**2 + self._y[self._subset_indices[1]]**2
            r11 = self._x[self._subset_indices[0] + 1]**2 + self._y[self._subset_indices[1] + 1]**2
            rmax = np.sqrt(max(r00.max(), r01.max(), r10.max(), r11.max()))
            rmin = np.sqrt(min(r00.min(), r01.min(), r10.min(), r11.min()))

            phi00 = np.atan2(self._y[self._subset_indices[1]], self._x[self._subset_indices[0]])
            phi01 = np.atan2(self._y[self._subset_indices[1] + 1], self._x[self._subset_indices[0]])
            phi10 = np.atan2(self._y[self._subset_indices[1]], self._x[self._subset_indices[0] + 1])
            phi11 = np.atan2(self._y[self._subset_indices[1] + 1], self._x[self._subset_indices[0] + 1])
            phimax = max(phi00.max(), phi01.max(), phi10.max(), phi11.max())
            phimin = min(phi00.min(), phi01.min(), phi10.min(), phi11.min())

            self._mesh_extent = {
                "xmin": self._x[imin], "xmax": self._x[imax],
                "ymin": self._y[jmin], "ymax": self._y[jmax],
                "zmin": self._z[kmin], "zmax": self._z[kmax],
                "rmin": rmin, "rmax": rmax,
                "phimin": phimin, "phimax": phimax,
            }

        # Calculate cell volume and centroid
        xc = 0.5 * (self._x[1:] + self._x[:-1])
        yc = 0.5 * (self._y[1:] + self._y[:-1])
        zc = 0.5 * (self._z[1:] + self._z[:-1])
        self._cell_centre = np.array([xc[self._subset_indices[0]],
                                      yc[self._subset_indices[1]],
                                      zc[self._subset_indices[2]]]).T

        dx = np.diff(self._x)[self._subset_indices[0]]
        dy = np.diff(self._y)[self._subset_indices[1]]
        dz = np.diff(self._z)[self._subset_indices[1]]

        if self._coordinate_system == 'cylindrical':
            self._cell_volume = dy * self._cell_centre[:, 0] * dx * dz
        elif self._coordinate_system == 'cartesian':
            self._cell_volume = dx * dy * dz
        
        self._cell_centre.setflags(write=False)
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
    def z(self):
        """Mesh z nodes."""
        return self._z
    
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
            value = np.array(value, dtype=np.int64)

            if value.ndim != 2:
                raise ValueError("Argument 'subset_indices' must be 2D array.")

            if value.shape[0] != 3:
                raise ValueError("Argument 'subset_indices' must be a (3, N)-shaped array.")

            ind1d = value[0] * self._z.size * self._y.size + value[1] * self._z.size + value[2]
            _, counts = np.unique(ind1d, return_counts=True)
            if np.any(counts > 1):
                raise ValueError("Argument 'subset_indices' must contain only unique indices.")
            self._subset_indices = (value[0].astype(np.int32), value[1].astype(np.int32), value[2].astype(np.int32))
        else:
            i, j, k = np.meshgrid(np.arange(self._x.size - 1, dtype=np.int32),
                                  np.arange(self._y.size - 1, dtype=np.int32),
                                  np.arange(self._z.size - 1, dtype=np.int32),
                                  indexing='ij')
            self._subset_indices = (i.flatten(), j.flatten(), k.flatten())

        self._subset_indices[0].setflags(write=False)
        self._subset_indices[1].setflags(write=False)
        self._subset_indices[2].setflags(write=False)
    
    def subset(self, indices, name=None):
        """
        Creates a subset StructGrid3D from this instance.
    
        If indices is the 2D (3, K)-shaped tuple, the subset is created with respect
        to the entire grid.
        If indices is the 1D array, the subset is created with respect
        to the current subset.

        :param indices: Indices of the cells with respect to the entire grid
                        or to the current subset.
        :param name: Name of the grid subset. Default is instance.name + ' subset'.
        """

        grid = StructGrid3D.__new__(StructGrid3D)

        grid._name = name or self.name + ' subset'
        grid._coordinate_system = self._coordinate_system
        grid._dimension = self._dimension
        grid._x = self._x
        grid._y = self._y
        grid._z = self._z

        indices = np.array(indices, dtype=np.int64)

        if indices.ndim == 1:
            # Create a new subset from the current subset
            _, counts = np.unique(indices, return_counts=True)
            if np.any(counts > 1):
                raise ValueError("Argument 'indices' must contain only unique indices.")

            if indices.max() >= self._num_cell:
                raise ValueError("The new subset is larger than the current subset.")
            
            grid._subset_indices = (self._subset_indices[0][indices],
                                    self._subset_indices[1][indices],
                                    self._subset_indices[2][indices])
            grid._initial_setup()

        elif indices.ndim == 3:
            # Create a subset from the entire grid
            grid.subset_indices = (indices[0].astype(np.int32),
                                   indices[1].astype(np.int32),
                                   indices[2].astype(np.int32))

        else:
            raise ValueError("Argument 'indices' must be 1D or 3D array.")

        return grid
    
    def interpolator(self, grid_data, fill_value=0):
        """
        Returns the StructGridFunction3D interpolator that takes the Cartesian coordinates.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_data: An array containing data in the grid cells.
        :param fill_value: A value returned outside the gird. Default is 0.

        :returns: StructGridFunction3D interpolator
        """
        if self._interpolator is None:
            self._interpolator = StructGridFunction3D(self._x, self._y, self._z, self._subset_indices, grid_data, fill_value)

            return self._interpolator
        
        instance = StructGridFunction3D.instance(self._interpolator, grid_data, fill_value)
        
        return instance

    def vector_interpolator(self, grid_vectors, fill_vector=Vector3D(0, 0, 0)):
        """
        Returns the StructGridVectorFunction3D interpolator.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_vectors: A (3,K) array containing 3D vectors in the grid cells.
        :param fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).

        :returns: StructGridVectorFunction3D interpolator
        """
        if self._interpolator is None:
            self._interpolator = StructGridVectorFunction3D(self._x, self._y, self._z, self._subset_indices, grid_vectors, fill_vector)

            return self._interpolator
        
        instance = StructGridVectorFunction3D.instance(self._interpolator, grid_vectors, fill_vector)
        
        return instance

    def __getstate__(self):
        state = {
            'name': self._name,
            'dimension': self._dimension,
            'coordinate_system': self._coordinate_system,
            'x': self._x,
            'y': self._y,
            'z': self._z,
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
        self._z = state['z']
        self._z.setflags(write=False)
        self._subset_indices = tuple(state['subset_indices'])
        self._subset_indices[0].setflags(write=False)
        self._subset_indices[1].setflags(write=False)
        self._subset_indices[2].setflags(write=False)

        self._initial_setup()
