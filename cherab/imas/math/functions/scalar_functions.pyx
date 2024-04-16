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

from raysect.core.math.function.float cimport autowrap_function1d, autowrap_function2d


cdef class ConstantMapper2D(Function2D):
    """
    Evaluates the given Function1D assuming that the function is
    a constant along the remaining axis.

    :param object function1d: 1D function to evaluate.
    :param object axis: The axis along which the function is constant.
        Must be ['x', 'y'] or [0, 1].
    """

    def __init__(self, object function1d, object axis):

        if isinstance(axis, str):
            map = {'x': 0, 'y': 1}
            try:
                axis = map[axis.lower()]
            except KeyError:
                raise ValueError("The axis must be either the string 'x', 'y', or the value 0 or 1.")

        # check numerical value
        if axis not in [0, 1]:
            raise ValueError("The axis must be either the string 'x', 'y', or the value 0 or 1.")

        self.axis = axis
        self._function = autowrap_function1d(function1d)

    cdef double evaluate(self, double x, double y) except? -1e999:

        if self.axis == 0:
            return self._function.evaluate(y)

        return self._function.evaluate(x)


cdef class ConstantMapper3D(Function3D):
    """
    Evaluates the given Function2D assuming that the function is
    a constant along the given axis.

    :param object function2d: 2D function to evaluate.
    :param object axis: The axis along which the function is constant.
        Must be ['x', 'y', 'z'] or [0, 1, 2].
    """

    def __init__(self, object function2d, object axis):

        if isinstance(axis, str):
            map = {'x': 0, 'y': 1, 'z': 2}
            try:
                axis = map[axis.lower()]
            except KeyError:
                raise ValueError("The axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")

        # check numerical value
        if axis not in [0, 1, 2]:
            raise ValueError("The axis must be either the string 'x', 'y' or 'z', or the value 0, 1 or 2.")

        self.axis = axis
        self._function = autowrap_function2d(function2d)

    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        if self.axis == 0:
            return self._function.evaluate(y, z)
        if self.axis == 1:
            return self._function.evaluate(x, z)

        return self._function.evaluate(x, y)

