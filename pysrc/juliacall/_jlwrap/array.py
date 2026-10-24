"""Julia wrappers initialized by PythonCall."""

import juliacall as jl
from juliacall import JlBase
from .collection import JlCollection

# Populated by Julia during initialization.
_jl_methods = {}


class JlArray(JlCollection):
    __module__ = "juliacall"
    __slots__ = ()
    _jl_buffer_info = None
    def __init__(self, value):
        JlBase.__init__(self, value, jl.Base.AbstractArray)
    @property
    def ndim(self):
        return self._jl_callmethod(_jl_methods["ndim"])
    @property
    def shape(self):
        return self._jl_callmethod(_jl_methods["shape"])
    def reshape(self, shape):
        return self._jl_callmethod(_jl_methods["reshape"], shape)
    def __getitem__(self, k):
        return self._jl_callmethod(_jl_methods["__getitem__"], k)
    def __setitem__(self, k, v):
        self._jl_callmethod(_jl_methods["__setitem__"], k, v)
    def __delitem__(self, k):
        self._jl_callmethod(_jl_methods["__delitem__"], k)
    @property
    def __array_interface__(self):
        return self._jl_callmethod(_jl_methods["__array_interface__"])
    def __array__(self, dtype=None, copy=None):
        import numpy
        # convert to an array-like object
        arr = self
        if not (hasattr(arr, "__array_interface__") or hasattr(arr, "__array_struct__")):
            if copy is False:
                raise ValueError("copy=False is not supported when collecting ArrayValue data")
            # the first attempt collects into an Array
            arr = self._jl_callmethod(_jl_methods["__array__"])
            if not (hasattr(arr, "__array_interface__") or hasattr(arr, "__array_struct__")):
                # the second attempt collects into a PyObjectArray
                arr = self._jl_callmethod(_jl_methods["__array___2"])
        # convert to a numpy array if numpy is available
        return numpy.array(arr, dtype=dtype, copy=copy)
    def to_numpy(self, dtype=None, copy=True, order="K"):
        import numpy
        return numpy.array(self, dtype=dtype, copy=copy, order=order)
