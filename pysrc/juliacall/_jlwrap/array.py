"""Julia value wrappers from src/JlWrap/array.jl."""

from .any import AnyValue

# Populated by Julia during PythonCall initialization.
_jl_methods = {}


class ArrayValue(AnyValue):
    __module__ = "juliacall"
    __slots__ = ()
    @property
    def _jl_buffer_info(self):
        return _jl_methods["_jl_buffer_info"]
    @property
    def ndim(self):
        return self._jl_callmethod(_jl_methods["ndim"])
    @property
    def shape(self):
        return self._jl_callmethod(_jl_methods["shape"])
    def copy(self):
        return self._jl_callmethod(_jl_methods["copy"])
    def reshape(self, shape):
        return self._jl_callmethod(_jl_methods["reshape"], shape)
    def __bool__(self):
        return bool(len(self))
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
                arr = self._jl_callmethod(_jl_methods["__array_pyobjectarray__"])
        # convert to a numpy array if numpy is available
        return numpy.array(arr, dtype=dtype, copy=copy)
    def to_numpy(self, dtype=None, copy=True, order="K"):
        import numpy
        return numpy.array(self, dtype=dtype, copy=copy, order=order)
