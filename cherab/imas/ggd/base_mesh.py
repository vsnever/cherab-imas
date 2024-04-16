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

from cherab.core.math import AxisymmetricMapper, VectorAxisymmetricMapper, CylindricalTransform, VectorCylindricalTransform
from cherab.imas.math import ConstantMapper3D, VectorConstantMapper3D

class GGDGrid:
    """
    Base class for general grids (GGD).

    :param str name: A name of the grid. Default is ''.
    :param str dimension: Grid dimensions. Default is 0.
    :param str coordinate_system: Coordinate system: 'cartesian' (default) or 'cylindrical'.

    :ivar num_cell: The number of grid cells.
    :ivar cell_centre: Coordinates of cell centres as (num_cell, dimension) array.
    :ivar cell_area: Cell areas as (num_cell,) array in 2D case.
    :ivar cell_volume: Cell volumes as (num_cell,) array in 3D case.
    :ivar mesh_extent: Extent of the mesh. A dictionary with xmin, xmax, ymin, ymax, ... keys.
    """

    def __init__(self, name='', dimension=1, coordinate_system='cartesian'):

        dimension = int(dimension)
        if dimension < 1:
            raise ValueError("Attribute dimension must be >= 1.")

        self._dimension = dimension
        self._name = str(name)

        coordinate_system = str(coordinate_system).lower()
        if coordinate_system not in ('cartesian', 'cylindrical'):
            raise ValueError("Only 'cartesian' and 'cylindrical' coordinate systems are supported.")

        self._coordinate_system = coordinate_system

        self._interpolator = None
        self._cell_centre = None
        self._cell_area = None
        self._cell_volume = None
        self._mesh_extent = None
        self._num_cell = 0

        self._initial_setup()

    def _initial_setup(self):

        raise NotImplementedError("To be defined in subclass.")
    
    @property
    def name(self):
        """Grid name."""
        return self._name
    
    @name.setter
    def name(self, value):
        self._name = str(value)
    
    @property
    def dimension(self):
        """Grid dimension."""
        return self._dimension
    
    @property
    def num_cell(self):
        """Number of grid cells."""
        return self._num_cell
    
    @property
    def coordinate_system(self):
        """Coordinate system."""
        return self._coordinate_system

    @property
    def cell_centre(self):
        """Coordinates of cell centres as (num_cell, dimension) array."""
        return self._cell_centre

    @property
    def cell_area(self):
        """Cell areas as (num_cell,) array."""
        return self._cell_area
    
    @property
    def cell_volume(self):
        """Cell volumes as (num_cell,) array."""
        return self._cell_volume

    @property
    def mesh_extent(self):
        """Extent of the mesh. A dictionary with xmin, xmax, ymin, ymax, ... keys."""
        return self._mesh_extent
    
    def subset(self, indices, name=None):
        """
        Creates a subset grid from this instance.

        :param indices: Indices of the cells of the original grid in the subset.
        :param name: Name of the grid subset. Default is instance.name + ' subset'.
        """

        raise NotImplementedError("To be defined in subclass.")
    
    def interpolator(self, grid_data, fill_value=0):
        """
        Returns an FunctionND interpolator instance for the data defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_data: An array containing data in the grid cells.
        :param fill_value: A value returned outside the gird. Default is 0.

        :returns: FunctionND interpolator
        """
    
        raise NotImplementedError("To be defined in subclass.")

    def vector_interpolator(self, grid_vectors, fill_vector=Vector3D(0, 0, 0)):
        """
        Returns a VectorFunctionND interpolator instance for the vector data
        defined on this grid.

        On the second and subsequent calls, the interpolator is created as an instance
        of the previously created interpolator.

        :param grid_vectors: A (num_cell, 3) array containing 3D vectors in the grid cells.
        :param fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).

        :returns: VectorFunctionND interpolator
        """

        raise NotImplementedError("To be defined in subclass.")

    def cartesian_3d_interpolator(self, grid_data, fill_value=0):
        """
        Returns an Function3D Cartesian interpolator instance for the data defined on this grid.
        In case of Cartesian 3D grids, the same as grid.interpolator.

        :param grid_data: An array containing data in the grid cells.
        :param fill_value: A value returned outside the gird. Default is 0.

        :returns: Function3D Cartesian interpolator
        """
        interp = self.interpolator(grid_data, fill_value)
    
        if self._dimension == 2:
            if self._coordinate_system == 'cylindrical':
                return AxisymmetricMapper(interp)
            
            return ConstantMapper3D(interp, axis=2)
        
        if self._dimension == 3:
            if self._coordinate_system == 'cylindrical':
                return CylindricalTransform(interp)
            
            return interp

    def cartesian_3d_vector_interpolator(self, grid_vectors, fill_vector=Vector3D(0, 0, 0)):
        """
        Returns a VectorFunction3D Cartesian interpolator instance for the vector data
        defined on this grid.
        In case of Cartesian 3D grids, the same as grid.vector_interpolator.

        :param grid_vectors: A (3,K) array containing 3D vectors in the grid cells.
        :param fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).

        :returns: Function3D Cartesian interpolator
        """
        interp = self.vector_interpolator(grid_vectors, fill_vector)

        if self._dimension == 2:
            if self._coordinate_system == 'cylindrical':
                return VectorAxisymmetricMapper(interp)
        
            return VectorConstantMapper3D(interp, axis=2)
        
        if self._dimension == 3:
            if self._coordinate_system == 'cylindrical':
                return VectorCylindricalTransform(interp)
            
            return interp

    def plot_mesh(self, data=None, ax=None):
        """
        Plot the grid geometry to a matplotlib figure.

        :param data: Data array defined on the grid.
        """

        raise NotImplementedError("To be defined in subclass.")
