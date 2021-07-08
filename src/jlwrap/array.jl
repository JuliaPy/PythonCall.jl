const pyjlarraytype = pynew()

function pyjlarray_array_interface(x::AbstractArray)
    # TODO
    errset(pybuiltins.AttributeError, "__array_interface__")
    return pynew()
end

function init_jlwrap_array()
    jl = pyjuliacallmodule
    filename = "$(@__FILE__):$(1+@__LINE__)"
    pybuiltins.exec(pybuiltins.compile("""
    class ArrayValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
        @property
        def ndim(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ ndims)))
        @property
        def shape(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ size)))
        def copy(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ copy)))
        @property
        def __array_interface__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlarray_array_interface)))
        def __array__(self):
            # convert to an array-like object
            if hasattr(self, "__array_interface__") or hasattr(self, "__array_struct__"):
                arr = self
            else:
                arr = self._jl_callmethod($(pyjl_methodnum(pyjl ∘ PyObjectArray)))
            # convert to a numpy array if numpy is available
            try:
                import numpy
                arr = numpy.array(arr)
            except ImportError:
                pass
            return arr
    """, filename, "exec"), jl.__dict__)
    pycopy!(pyjlarraytype, jl.ArrayValue)
end

pyjl(v::AbstractArray) = pyjl(pyjlarraytype, v)
