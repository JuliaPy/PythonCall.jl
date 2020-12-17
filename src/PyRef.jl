"""
    PyRef([x])

A reference to a Python object converted from `x`, or a null reference if `x` is not given.

This is baically just a mutable reference to a pointer to a Python object.
It owns the reference (if non-NULL) and automatically decrefs it when finalized.

Building block for more complex wrapper types such as `PyObject` and `PyList`.
"""
mutable struct PyRef
    ptr :: CPyPtr
    PyRef(::Val{:new}, ptr::Ptr, borrowed::Bool) = begin
        x = new(CPyPtr(ptr))
        borrowed && C.Py_IncRef(ptr)
        finalizer(x) do x
            if !CONFIG.isinitialized
                ptr = x.ptr
                if !isnull(ptr)
                    with_gil(false) do
                        C.Py_DecRef(ptr)
                    end
                    x.ptr = CPyPtr()
                end
            end
        end
        x
    end
end
export PyRef

pynewref(x::Ptr, check::Bool=false) = (check && isnull(x)) ? pythrow() : PyRef(Val(:new), x, false)
pyborrowedref(x::Ptr, check::Bool=false) = (check && isnull(x)) ? pythrow() : PyRef(Val(:new), x, true)

ispyreftype(::Type{PyRef}) = true
pyptr(x::PyRef) = x.ptr
isnull(x::PyRef) = isnull(pyptr(x))
Base.unsafe_convert(::Type{CPyPtr}, x::PyRef) = pyptr(x)
C.PyObject_TryConvert__initial(o, ::Type{PyRef}) = C.putresult(PyRef, pyborrowedref(o))

PyRef(x) = begin
    ptr = C.PyObject_From(x)
    isnull(ptr) && pythrow()
    pynewref(ptr)
end
PyRef() = pynewref(CPyPtr())

Base.convert(::Type{PyRef}, x::PyRef) = x
Base.convert(::Type{PyRef}, x) = PyRef(x)
