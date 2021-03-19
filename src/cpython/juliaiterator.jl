mutable struct Iterator
    val::Any
    st::Any
end
Iterator(x) = Iterator(x, nothing)
Base.length(x::Iterator) = length(x.val)

pyjliter_iter(xo::PyPtr) =
    PyJuliaIteratorValue_New(Iterator((PyJuliaValue_GetValue(xo)::Iterator).val))

pyjliter_iternext(xo::PyPtr) =
    try
        x = PyJuliaValue_GetValue(xo)::Iterator
        val = x.val
        st = x.st
        if st === nothing
            z = iterate(val)
        else
            z = iterate(val, something(st))
        end
        if z === nothing
            PyNULL
        else
            r, newst = z
            x.st = Some(newst)
            PyObject_From(r)
        end
    catch err
        if err isa MethodError && err.f === iterate
            PyErr_SetStringFromJuliaError(PyExc_TypeError(), err)
        else
            PyErr_SetJuliaError(err)
        end
        PyNULL
    end

const PyJuliaIteratorValue_Type = LazyPyObject() do
    c = []
    base = PyJuliaAnyValue_Type()
    isnull(base) && return PyNULL
    ptr = PyPtr(cacheptr!(c, fill(PyTypeObject(
        name = cacheptr!(c, "juliacall.IteratorValue"),
        base = base,
        iter = @cfunctionOO(pyjliter_iter),
        iternext = @cfunctionOO(pyjliter_iternext),
    ))))
    err = PyType_Ready(ptr)
    ism1(err) && return PyNULL
    PYJLGCCACHE[ptr] = c
    return ptr
end

PyJuliaIteratorValue_New(x::Iterator) = PyJuliaValue_New(PyJuliaIteratorValue_Type(), x)
