const PyJuliaNumberValue_Type__ref = Ref(PyPtr())
PyJuliaNumberValue_Type() = begin
    ptr = PyJuliaNumberValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaAnyValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.NumberValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyNumberABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaNumberValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaNumberValue_New(x::Number) = PyJuliaValue_New(PyJuliaNumberValue_Type(), x)
PyJuliaValue_From(x::Number) = PyJuliaNumberValue_New(x)

const PyJuliaComplexValue_Type__ref = Ref(PyPtr())
PyJuliaComplexValue_Type() = begin
    ptr = PyJuliaComplexValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaNumberValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.ComplexValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyComplexABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaComplexValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaComplexValue_New(x::Complex) = PyJuliaValue_New(PyJuliaComplexValue_Type(), x)
PyJuliaValue_From(x::Complex) = PyJuliaComplexValue_New(x)

const PyJuliaRealValue_Type__ref = Ref(PyPtr())
PyJuliaRealValue_Type() = begin
    ptr = PyJuliaRealValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaNumberValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.RealValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyRealABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaRealValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaRealValue_New(x::Real) = PyJuliaValue_New(PyJuliaRealValue_Type(), x)
PyJuliaValue_From(x::Real) = PyJuliaRealValue_New(x)

const PyJuliaRationalValue_Type__ref = Ref(PyPtr())
PyJuliaRationalValue_Type() = begin
    ptr = PyJuliaRationalValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaRealValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.RationalValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyRationalABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaRationalValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaRationalValue_New(x::Rational) = PyJuliaValue_New(PyJuliaRationalValue_Type(), x)
PyJuliaValue_From(x::Rational) = PyJuliaRationalValue_New(x)

const PyJuliaIntegerValue_Type__ref = Ref(PyPtr())
PyJuliaIntegerValue_Type() = begin
    ptr = PyJuliaIntegerValue_Type__ref[]
    if isnull(ptr)
        c = []
        base = PyJuliaRealValue_Type()
        isnull(base) && return PyPtr()
        t = fill(PyType_Create(c,
            name = "julia.IntegerValue",
            base = base,
        ))
        ptr = PyPtr(pointer(t))
        err = PyType_Ready(ptr)
        ism1(err) && return PyPtr()
        abc = PyIntegralABC_Type()
        isnull(abc) && return PyPtr()
        ism1(PyABC_Register(ptr, abc)) && return PyPtr()
        PYJLGCCACHE[ptr] = push!(c, t)
        PyJuliaIntegerValue_Type__ref[] = ptr
    end
    ptr
end

PyJuliaIntegerValue_New(x::Integer) = PyJuliaValue_New(PyJuliaIntegerValue_Type(), x)
PyJuliaValue_From(x::Integer) = PyJuliaIntegerValue_New(x)
