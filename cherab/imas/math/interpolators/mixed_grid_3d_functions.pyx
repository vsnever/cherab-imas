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
from raysect.core.math.point cimport new_point2d
from raysect.core.math.cython.utility cimport find_index

cimport cython


cdef class MixedGridFunction3D(Function3D):
    """
    A simple interpolator for the data defined on the 3D mixed grid,
    the 2D unstructured grid extended structuraly along the third axis.
    Finds the cell containing the point (x, y, z). Uses 2D KDtree algorithm.
    Returns the data value for this cell or the `fill_value` if the grid
    does not contain the point. 

    :param object vertex_coords: 2D (N,3) array-like with the vertex coordinates of triangles
        in the plane.
    :param object triangles: 2D (M,3) integer array-like with the vertex indices forming
        the triangles.
    :param object z: The L corners of the rectangular cells along the third axis.
    :param object triangle_to_cell_map: 1D (M,) integer array-like with the indices of
        the polygons in the plane containing the triangles.
    :param ndarray grid_data: A (K,L-1)-shaped array containing data in the grid cells.
        First dimension for polygons in the plane, second dimension for cells along the third axis.
    :param object z_axis: The axis along which the 2D unstructured grid is structuraly extended in 3D.
        Must be ['x', 'y', 'z'] or [0, 1, 2].
    :param double fill_value: A value returned outside the grid. Default is 0.
    """

    def __init__(self, object vertex_coords not None, object triangles not None, object z not None,
                 object triangle_to_cell_map not None, np.ndarray grid_data not None,
                 object z_axis=2, double fill_value=0):
        
        if isinstance(z_axis, str):
            map = {'x': 0, 'y': 1, 'z': 2}
            try:
                z_axis = map[z_axis.lower()]
            except KeyError:
                raise ValueError("The z_axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")

        # check numerical value
        if z_axis not in [0, 1, 2]:
            raise ValueError("The z_axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")
        
        self.z_axis = z_axis

        self._z = np.array(z, dtype=np.float64)

        if self._z.ndim != 1:
            raise ValueError("Argument 'z' must be 1D array.")

        if self._z.size < 2:
            raise ValueError("Array 'z' must have at least 2 elements.")

        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        triangles = np.array(triangles, dtype=np.int32)
        triangle_to_cell_map = np.array(triangle_to_cell_map, dtype=np.int32)

        # build kdtree
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        self._triangle_to_cell_map = triangle_to_cell_map
       
        # Attention!!! Do not copy grid_data! Attribute self._grid_data must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        if grid_data.ndim != 2:
            raise ValueError("The grid_data must be a 2D array.")

        if grid_data.shape[1] != self._z - 1:
            raise ValueError("The shape of the grid_data array does not match the number of cells along z-axis.")

        self._grid_data = grid_data
        self._fill_value = fill_value

        self._z_mv = self._z
        self._triangle_to_cell_map_mv = self._triangle_to_cell_map
        self._grid_data_mv = self._grid_data

    def __getstate__(self):
        return self._grid_data, self._fill_value, self._triangle_to_cell_map, self._kdtree, self._z, self.z_axis

    def __setstate__(self, state):
        self._grid_data, self._fill_value, self._triangle_to_cell_map, self._kdtree, self._z, self.z_axis = state
        self._z_mv = self._z
        self._triangle_to_cell_map_mv = self._triangle_to_cell_map
        self._grid_data_mv = self._grid_data

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_data=None, object fill_value=None):
        """
        Creates a new interpolator instance from an existing MixedGridFunction3D
        or MixedGridVectorFunction3D instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_data of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        If created from the MixedGridVectorFunction3D instance,
        the grid_data and the fill_value must not be None.

        :param object instance: MixedGridFunction3D or MixedGridVectorFunction3D object.
        :param ndarray grid_data: A 2D array containing data in the grid cells.
        :param object fill_value: A value returned outside the grid.
        :rtype: MixedGridFunction3D
        """

        cdef MixedGridFunction3D m, inst
        cdef MixedGridVectorFunction3D instvec

        if grid_data.ndim != 2:
            raise ValueError("The grid_data must be a 2D array.")

        m = MixedGridFunction3D.__new__(MixedGridFunction3D)

        if isinstance(instance, MixedGridFunction3D):
            inst = instance
            # copy source data
            m._kdtree = inst._kdtree
            m._z = inst._z
            m.z_axis = inst.z_axis
            m._triangle_to_cell_map = inst._triangle_to_cell_map

            # replace grid data and fill value
            m._grid_data = inst._grid_data if grid_data is None else grid_data
            m._fill_value = inst._fill_value if fill_value is None else <double>fill_value
        elif isinstance(instance, MixedGridVectorFunction3D):
            instvec = instance
            m._kdtree = instvec._kdtree
            m._z = instvec._z
            m.z_axis = instvec.z_axis
            m._triangle_to_cell_map = instvec._triangle_to_cell_map

            if grid_data is None:
                raise ValueError("Argument 'grid_data' must not be None if the new instant MixedGridFunction3D is created from the MixedGridVectorFunction3D instance.")
            if fill_value is None:
                raise ValueError("Argument 'fill_value' must not be None if the new instant MixedGridFunction3D is created from the MixedGridVectorFunction3D instance.")
            m._grid_data = grid_data
            m._fill_value = <double>fill_value
        else:
            raise TypeError("Argument 'instance' must be either MixedGridFunction3D or MixedGridVectorFunction3D instance.")

        m._triangle_to_cell_map_mv = m._triangle_to_cell_map
        m._z_mv = m._z
        m._grid_data_mv = m._grid_data

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            np.int32_t triangle_id, ipoly, iz
            double t
        
        if self.z_axis == 0:
            t = x
            x = y
            y = z
            z = t
        elif self.z_axis == 1:
            t = y
            y = z
            z = t

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id
            ipoly = self._triangle_to_cell_map_mv[triangle_id]
            iz = find_index(self._z_mv, z)

            return self._grid_data_mv[ipoly, iz]

        return self._fill_value

cdef class MixedGridVectorFunction3D(VectorFunction3D):
    """
    A simple interpolator for the data defined on the 3D mixed grid,
    the 2D unstructured grid in the plane extended structuraly along the third axis.
    Finds the cell containing the point (x, y, z). Uses 2D KDtree algorithm in the plane.
    Returns the 3D vector value for this cell or the `fill_vector` if the grid
    does not contain the point.

    :param object vertex_coords: 2D (N,3) array-like with the vertex coordinates of triangles.
    :param object triangles: 2D (M,3) integer array-like with the vertex indices forming
        the triangles.
    :param object z: The L corners of the rectangular cells along the third axis.
    :param object triangle_to_cell_map: 1D (M,) integer array-like with the indices of
        the grid cells (polygons) containing the triangles.
    :param ndarray grid_vectors: A (3,K,L-1) array containing 3D vectors in the grid cells.
    :param object z_axis: The axis along which the 2D unstructured grid is structuraly extended in 3D.
        Must be ['x', 'y', 'z'] or [0, 1, 2].
    :param Vector3D fill_vector: A 3D vector returned outside the grid. Default is (0, 0, 0).
    """

    def __init__(self, object vertex_coords not None, object triangles not None, object z not None,
                 object triangle_to_cell_map not None, np.ndarray grid_vectors not None,
                 object z_axis=2, Vector3D fill_vector=Vector3D(0, 0, 0)):

        if isinstance(z_axis, str):
            map = {'x': 0, 'y': 1, 'z': 2}
            try:
                z_axis = map[z_axis.lower()]
            except KeyError:
                raise ValueError("The z_axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")

        # check numerical value
        if z_axis not in [0, 1, 2]:
            raise ValueError("The z_axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")
        
        self.z_axis = z_axis
        
        self._z = np.array(z, dtype=np.float64)

        if self._z.ndim != 1:
            raise ValueError("Argument 'z' must be 1D array.")

        if self._z.size < 2:
            raise ValueError("Array 'z' must have at least 2 elements.")

        vertex_coords = np.array(vertex_coords, dtype=np.float64)
        triangles = np.array(triangles, dtype=np.int32)
        triangle_to_cell_map = np.array(triangle_to_cell_map, dtype=np.int32)

        # build kdtree
        self._kdtree = MeshKDTree2D(vertex_coords, triangles)

        self._triangle_to_cell_map = triangle_to_cell_map

        # Attention!!! Do not copy grid_vectors! Attribute self._grid_vectors must point to the original data array,
        # so as not to re-initialize the interpolator if the user changes data values.

        if grid_vectors.ndim != 3:
            raise ValueError("The grid_vectors must be a 3D array.")

        if grid_vectors.shape[0] != 3:
            raise ValueError("The grid_vectors must be a (3,N,M)-shaped array.")

        if grid_vectors.shape[2] != self._z - 1:
            raise ValueError("The shape of the grid_vectors array does not match the number of cells along z-axis.")

        # populate internal attributes
        self._grid_vectors = grid_vectors
        self._fill_vector = fill_vector

        self._z_mv = self._z
        self._triangle_to_cell_map_mv = self._triangle_to_cell_map
        self._grid_vectors_mv = self._grid_vectors

    def __getstate__(self):
        return self._grid_vectors, self._fill_vector, self._triangle_to_cell_map, self._kdtree, self._z, self.z_axis

    def __setstate__(self, state):
        self._grid_vectors, self._fill_vector, self._triangle_to_cell_map, self._kdtre, self._z, self.z_axis = state
        self._grid_vectors_mv = self._grid_vectors
        self._z_mv = self._z
        self._triangle_to_cell_map_mv = self._triangle_to_cell_map

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @classmethod
    def instance(cls, object instance not None, np.ndarray grid_vectors=None,
                 Vector3D fill_vector=None):
        """
        Creates a new interpolator instance from an existing MixedGridVectorFunction3D
        or MixedGridFunction3D instance.
        The new interpolator instance will share the same internal acceleration
        data as the original interpolator. The grid_vectors of the new instance can
        be redefined.
        This method should be used if the user has multiple datasets
        that lie on the same mesh geometry. Using this methods avoids the
        repeated rebuilding of the mesh acceleration structures by sharing the
        geometry data between multiple interpolator objects.

        If created from the MixedGridFunction3D instance,
        the grid_vectors and the fill_vector must not be None.

        :param object instance: MixedGridVectorFunction3D or MixedGridFunction3D object.
        :param ndarray grid_vectors: An array containing vector grid data.
        :rtype: MixedGridVectorFunction3D
        """

        cdef MixedGridVectorFunction3D m, instvec
        cdef MixedGridFunction3D inst

        m = MixedGridVectorFunction3D.__new__(MixedGridVectorFunction3D)

        if isinstance(instance, MixedGridVectorFunction3D):
            instvec = instance
            # copy source data
            m._kdtree = instvec._kdtree
            m._z = instvec._z
            m.z_axis = instvec.z_axis
            m._triangle_to_cell_map = instvec._triangle_to_cell_map

            # replace grid vector and fill vector
            m._grid_vectors = instvec._grid_vectors if grid_vectors is None else grid_vectors
            m._fill_vector = instvec._fill_vector if fill_vector is None else fill_vector
        elif isinstance(instance, MixedGridFunction3D):
            inst = instance
            m._kdtree = inst._kdtree
            m._z = inst._z
            m.z_axis = inst.z_axis
            m._triangle_to_cell_map = inst._triangle_to_cell_map

            if grid_vectors is None:
                raise ValueError("Argument 'grid_vectors' must not be None if the new instant MixedGridVectorFunction3D is created from the MixedGridFunction3D instance.")
            if fill_vector is None:
                raise ValueError("Argument 'fill_vector' must not be None if the new instant MixedGridVectorFunction3D is created from the MixedGridFunction3D instance.")
            m._grid_vectors = grid_vectors
            m._fill_vector = fill_vector
        else:
            raise TypeError("Argument 'instance' must be either MixedGridFunction3D or MixedGridVectorFunction3D instance.")

        m._triangle_to_cell_map_mv = m._triangle_to_cell_map
        m._z_mv = m._z
        m._grid_vectors_mv = m._grid_vectors

        return m

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef Vector3D evaluate(self, double x, double y, double z):

        cdef:
            np.int32_t triangle_id, ipoly, iz
            double vx, vy, vz, t
        
        if self.z_axis == 0:
            t = x
            x = y
            y = z
            z = t
        elif self.z_axis == 1:
            t = y
            y = z
            z = t

        if self._kdtree.is_contained(new_point2d(x, y)):

            triangle_id = self._kdtree.triangle_id
            ipoly = self._triangle_to_cell_map_mv[triangle_id]
            iz = find_index(self._z_mv, z)
            vx = self._grid_vectors_mv[0, ipoly, iz]
            vy = self._grid_vectors_mv[1, ipoly, iz]
            vz = self._grid_vectors_mv[2, ipoly, iz]

            return new_vector3d(vx, vy, vz)

        return self._fill_vector
