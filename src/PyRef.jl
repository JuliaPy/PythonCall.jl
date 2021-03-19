"""
    PyRef([x])

A reference to a Python object converted from `x`, or a null reference if `x` is not given.

This is baically just a mutable reference to a pointer to a Python object.
It owns the reference (if non-NULL) and automatically decrefs it when finalized.

Building block for more complex wrapper types such as `PyObject` and `PyList`.
"""
mutable struct PyRef
    ptr::CPyPtr
    PyRef(::Val{:new}, ptr::Ptr, borrowed::Bool) = begin
        x = new(CPyPtr(ptr))
        borrowed && C.Py_IncRef(ptr)
        finalizer(x) do x
            if CONFIG.isinitialized
                ptr = x.ptr
                if !isnull(ptr)
                    with_gil(false) do
                        @assert C.Py_RefCnt(ptr) > 0
                        C.Py_DecRef(ptr)
                    end
                    x.ptr = C_NULL
                end
            end
        end
        x
    end
end
export PyRef

pynewref(x::Ptr, check::Bool = false) =
    (check && isnull(x)) ? pythrow() : PyRef(Val(:new), x, false)
pyborrowedref(x::Ptr, check::Bool = false) =
    (check && isnull(x)) ? pythrow() : PyRef(Val(:new), x, true)

ispyreftype(::Type{PyRef}) = true
pyptr(x::PyRef) = x.ptr
isnull(x::PyRef) = isnull(pyptr(x))
Base.unsafe_convert(::Type{CPyPtr}, x::PyRef) = pyptr(x)
C.PyObject_TryConvert__initial(o, ::Type{PyRef}) = C.putresult(pyborrowedref(o))

PyRef(x) = begin
    ptr = C.PyObject_From(x)
    isnull(ptr) && pythrow()
    pynewref(ptr)
end
PyRef() = pynewref(C_NULL)

Base.convert(::Type{PyRef}, x::PyRef) = x
Base.convert(::Type{PyRef}, x) = PyRef(x)

# Cache some common standard modules
for name in [
    "os",
    "io",
    "sys",
    "pprint",
    "collections",
    "collections.abc",
    "numbers",
    "fractions",
    "datetime",
    "numpy",
    "pandas",
]
    f = Symbol("py", replace(name, "." => ""), "module")
    rf = Symbol("_", f)
    @eval $rf = PyRef()
    @eval $f(::Type{T}) where {T} = begin
        r = $rf
        if isnull(r.ptr)
            m = pyimport(PyRef, $name)
            C.Py_IncRef(m.ptr)
            r.ptr = m.ptr
        end
        (r isa T) ? r : pyconvert(T, r)
    end
    @eval $f() = $f(PyObject)
end
