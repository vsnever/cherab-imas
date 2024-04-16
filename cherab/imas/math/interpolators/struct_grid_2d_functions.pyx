# cython: language_level=3

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

from raysect.core.math.vector cimport new_vector3d
from raysect.core.math.cython.utility cimport find_index

cimport cython


cdef class StructGridFunction2D(Function2D):
    """
    A simple interpolator for the data defined on the 2D structured grid.
    Finds the cell containing the point (x, y).
    Checks if the cell is contained within the grid subset.
    Returns the data value for this cell or the `fill_value` if the points lies
    outside the subset. 

    :param object x: The corners of the rectangular cells along x axis.
    :param object y: The corners of the rectangular cells along y axis.
    :param ndarray grid_data: A 1D N-size array with the data defined on the subset.
    :param object subset_indices: A (2, N)-shaped tuple containing cell indices
        of the grid subset. Default is None (the subset contains the whole grid).
    :param double fill_value: A value returned outside the subset. Default is 0.
    """

    def __init__(self, object x not None, object y not None, np.ndarray grid_data not None,
                 object subset_indices=None, double fill_value=0):

        self._x = np.array(x, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)

        if self._x.ndim != 1:
            raise ValueError("Argument 'x' must be 1D array.")
        if self._y.ndim != 1:
            raise ValueError("Argument 'y' must be 1D array.")

        if self._x.size < 2:
            raise ValueError("Array 'x' must have at least 2 elements.")
        if self._y.size < 2:
            raise ValueError("Array 'y' must have at least 2 elements.")

        if subset_indices is not None:
            subset_indices = np.array(subset_indices, dtype=np.int32)

            if subset_indices.ndim != 2:
                raise ValueError("Argument 'subset_indices' must be 2D array.")
            
            if subset_indices.shape[0] != 2:
                raise ValueError("Argument 'subset_indices' must be a (2, N)-shaped array.")
            
            self._subset_map = -1 * np.ones((self._x.size - 1, self._y.size - 1), dtype=np.int32)         
            self._subset_size = subset_indices.shape[1]       
            self._subset_map[(subset_indices[0], subset_indices[1])] = np.arange(self._subset_size, dtype=np.int32)
        
        else:
            self._subset_size = (self._x.size - 1) * (self._y.size - 1)
            self._subset_map = np.arange(self._subset_size, dtype=np.int32).reshape((self._x.size - 1, self._y.size - 1))

        # Attention!!! Do not copy grid_data! Attribute self._grid_data must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        if grid_data.ndim != 1:
            raise ValueError("The grid_data must be a 1D array.")

        if grid_data.size != self._subset_size:
            raise ValueError("The size of the grid_data array does not match the number of cells in the subset.")

        self._grid_data = grid_data
        self._fill_value = fill_value

        self._x_mv = self._x
        self._y_mv = self._y
        self._subset_map_mv = self._subset_map
        self._grid_data_mv = self._grid_data

    def __getstate__(self):
        return self._grid_data, self._subset_map, self._fill_value, self._x, self._y, self._subset_size

    def __setstate__(self, state):
        self._grid_data, self._subset_map, self._fill_value, self._x, self._y, self._subset_size = state
        self._x_mv = self._x
        self._y_mv = self._y
        self._subset_map_mv = self._subset_map
        self._grid_data_mv = self._grid_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_data=None, object fill_value=None):
        """
        Creates a new interpolator instance from an existing StructGridFunction2D
        or StructGridVectorFunction2D instance.
        The new interpolator instance will share the subset-to-grid map with
        the original interpolator. The grid_data of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same subset of the grid. Using this methods reduces memory usage.

        If created from the StructGridVectorFunction2D instance,
        the grid_data and the fill_value must not be None.

        :param object instance: StructGridFunction2D or StructGridVectorFunction2D object.
        :param ndarray grid_data: An array containing data in the grid cells.
        :param object fill_value: A value returned outside the gird.
        :rtype: StructGridFunction2D
        """

        cdef StructGridFunction2D m, inst
        cdef StructGridVectorFunction2D instvec

        m = StructGridFunction2D.__new__(StructGridFunction2D)

        if isinstance(instance, StructGridFunction2D):
            inst = instance
            # copy source data
            m._x = inst._x
            m._y = inst._y
            m._subset_map = inst._subset_map

            # replace grid data and fill value
            m._grid_data = inst._grid_data if grid_data is None else grid_data
            m._fill_value = inst._fill_value if fill_value is None else <double>fill_value
        elif isinstance(instance, StructGridVectorFunction2D):
            instvec = instance
            m._x = instvec._x
            m._y = instvec._y        
            m._subset_map = instvec._subset_map

            if grid_data is None:
                raise ValueError("Argument 'grid_data' must not be None if the new instant StructGridFunction2D is created from the StructGridVectorFunction2D instance.")
            if fill_value is None:
                raise ValueError("Argument 'fill_value' must not be None if the new instant StructGridFunction2D is created from the StructGridVectorFunction2D instance.")
            m._grid_data = grid_data
            m._fill_value = <double>fill_value
        else:
            raise TypeError("Argument 'instance' must be either StructGridFunction2D or StructGridVectorFunction2D instance.")

        m._x_mv = m._x
        m._y_mv = m._y
        m._subset_map_mv = m._subset_map
        m._grid_data_mv = m._grid_data

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y) except? -1e999:

        cdef int ix = find_index(self._x_mv, x)
        cdef int iy = find_index(self._y_mv, y)
        cdef int indx

        if -1 < ix < self._x_mv.shape[0] and -1 < iy < self._y_mv.shape[0]:
            indx = self._subset_map_mv[ix, iy]
            if indx > -1:
                return self._grid_data_mv[indx]

        return self._fill_value


cdef class StructGridVectorFunction2D(VectorFunction2D):
    """
    A simple vector interpolator for the data defined on the 2D structured grid.
    Finds the cell containing the point (x, y).
    Checks if the cell is contained within the grid subset.
    Returns the 3D vector value for this cell or the `fill_vector` if the points lies
    outside the subset.

    :param object x: The corners of the quadrilateral cells along x axis.
    :param object y: The corners of the quadrilateral cells along y axis.
    :param ndarray grid_vectors: An (3, N)-shaped array containing 3D vectors
        on the grid subset.
    :param object subset_indices: A (2, N)-shaped array-like containing cell indices
        of the grid subset. Default is None (the subset contains the whole grid).
    :param Vector3D fill_vector: A 3D vector returned outside the gird. Default is (0, 0, 0).
    """

    def __init__(self, object x not None, object y not None, np.ndarray grid_vectors not None,
                 object subset_indices=None, Vector3D fill_vector=Vector3D(0, 0, 0)):

        self._x = np.array(x, dtype=np.float64)
        self._y = np.array(y, dtype=np.float64)

        if self._x.ndim != 1:
            raise ValueError("Argument 'x' must be 1D array.")
        if self._y.ndim != 1:
            raise ValueError("Argument 'y' must be 1D array.")

        if self._x.size < 2:
            raise ValueError("Array 'x' must have at least 2 elements.")
        if self._y.size < 2:
            raise ValueError("Array 'y' must have at least 2 elements.")
        
        if subset_indices is not None:
            subset_indices = np.array(subset_indices, dtype=np.int32)

            if subset_indices.ndim != 2:
                raise ValueError("Argument 'subset_indices' must be 2D array.")
            
            if subset_indices.shape[0] != 2:
                raise ValueError("Argument 'subset_indices' must be a (2, N)-shaped array.")
            
            self._subset_map = -1 * np.ones((self._x.size - 1, self._y.size - 1), dtype=np.int32)         
            self._subset_size = subset_indices.shape[1]       
            self._subset_map[(subset_indices[0], subset_indices[1])] = np.arange(self._subset_size, dtype=np.int32)
        
        else:
            self._subset_size = (self._x.size - 1) * (self._y.size - 1)
            self._subset_map = np.arange(self._subset_size, dtype=np.int32).reshape((self._x.size - 1, self._y.size - 1))

        # Attention!!! Do not copy grid_vectors! Attribute self._grid_vectors must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        if grid_vectors.ndim != 2:
            raise ValueError("The grid_vectors must be a 2D array.")

        if grid_vectors.shape[0] != 3:
            raise ValueError("The grid_vectors must be a (3, N)-shaped array.")

        if grid_vectors.shape[1] != self._subset_size:
            raise ValueError("The grid_vectors.shape[1] does not match the number of cells in the subset.")

        self._grid_vectors = grid_vectors
        self._fill_vector = fill_vector

        self._x_mv = self._x
        self._y_mv = self._y
        self._subset_map_mv = self._subset_map
        self._grid_vectors_mv = self._grid_vectors

    def __getstate__(self):
        return self._grid_vectors, self._subset_map, self._fill_vector, self._x, self._y, self._subset_size

    def __setstate__(self, state):
        self._grid_vectors, self._subset_map, self._fill_vector, self._x, self._y, self._subset_size = state
        self._x_mv = self._x
        self._y_mv = self._y
        self._subset_map_mv = self._subset_map
        self._grid_vectors_mv = self._grid_vectors

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_vectors=None,
                 Vector3D fill_vector=None):
        """
        Creates a new interpolator instance from an existing StructGridVectorFunction2D
        or StructGridFunction2D instance.
        The new interpolator instance will share the subset-to-grid map with
        the original interpolator. The grid_vectors of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same subset of the grid. Using this methods reduces memory usage.

        If created from the StructGridFunction2D instance,
        the grid_vectors and the fill_vector must not be None.

        :param object instance: StructGridVectorFunction2D or StructGridFunction2D object.
        :param ndarray grid_vectors: An array containing vector data in the grid cells.
        :param object fill_value: A value returned outside the gird.
        :rtype: StructGridVectorFunction2D
        """

        cdef StructGridVectorFunction2D m, instvec
        cdef StructGridFunction2D inst

        m = StructGridVectorFunction2D.__new__(StructGridVectorFunction2D)

        if isinstance(instance, StructGridVectorFunction2D):
            instvec = instance
            # copy source data
            m._x = instvec._x
            m._y = instvec._y
            m._subset_map = instvec._subset_map

            # replace grid data and fill value
            m._grid_vectors = instvec._grid_vectors if grid_vectors is None else grid_vectors
            m._fill_vector = instvec._fill_vector if fill_vector is None else fill_vector
        elif isinstance(instance, StructGridFunction2D):
            inst = instance
            m._x = inst._x
            m._y = inst._y    
            m._subset_map = inst._subset_map

            if grid_vectors is None:
                raise ValueError("Argument 'grid_vectors' must not be None if the new instant StructGridVectorFunction2D is created from the StructGridFunction2D instance.")
            if fill_vector is None:
                raise ValueError("Argument 'fill_vector' must not be None if the new instant StructGridVectorFunction2D is created from the StructGridFunction2D instance.")
            m._grid_vectors = grid_vectors
            m._fill_vector = fill_vector
        else:
            raise TypeError("Argument 'instance' must be either StructGridFunction2D or StructGridVectorFunction2D instance.")

        m._x_mv = m._x
        m._y_mv = m._y
        m._subset_map_mv = m._subset_map
        m._grid_vectors_mv = m._grid_vectors

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y):

        cdef int ix = find_index(self._x_mv, x)
        cdef int iy = find_index(self._y_mv, y)
        cdef double vx, vy, vz
        cdef int indx

        if -1 < ix < self._x_mv.shape[0] and -1 < iy < self._y_mv.shape[0]:

            indx = self._subset_map_mv[ix, iy]
            if indx > -1:
                vx = self._grid_vectors_mv[0, indx]
                vy = self._grid_vectors_mv[1, indx]
                vz = self._grid_vectors_mv[2, indx]

                return new_vector3d(vx, vy, vz)

        return self._fill_vector
