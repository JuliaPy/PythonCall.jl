mutable struct Iterator
    val::Any
    st::Any
end
Iterator(x) = Iterator(x, nothing)
Base.length(x::Iterator) = length(x.val)

const PyJuliaIteratorValue_Type__ref = Ref(PyPtr())
PyJuliaIteratorValue_Type() = begin
    ptr = PyJuliaIteratorValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(
            PyType_Create(
                c,
                name = "julia.IteratorValue",
                base = base,
                iter = pyjliter_iter,
                iternext = pyjliter_iternext,
            ),
        )
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaIteratorValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaIteratorValue_New(x::Iterator) = PyJuliaValue_New(PyJuliaIteratorValue_Type(), x)

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
            PyPtr()
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
        PyPtr()
    end
