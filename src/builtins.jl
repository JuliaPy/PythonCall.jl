cpyop(f::Function, x) = begin
    xo = C.PyObject_From(x)
    isnull(xo) && pythrow()
    r = f(xo)
    C.Py_DecRef(xo)
    r
end

cpyop(f::Function, x, y) = begin
    xo = C.PyObject_From(x)
    isnull(xo) && pythrow()
    yo = C.PyObject_From(y)
    isnull(yo) && (C.Py_DecRef(xo); pythrow())
    r = f(xo, yo)
    C.Py_DecRef(xo)
    C.Py_DecRef(yo)
    r
end

cpyop(f::Function, x, y, z) = begin
    xo = C.PyObject_From(x)
    isnull(xo) && pythrow()
    yo = C.PyObject_From(y)
    isnull(yo) && (C.Py_DecRef(xo); pythrow())
    zo = C.PyObject_From(z)
    isnull(yo) && (C.Py_DecRef(xo); C.Py_DecRef(yo); pythrow())
    r = f(xo, yo, zo)
    C.Py_DecRef(xo)
    C.Py_DecRef(yo)
    C.Py_DecRef(zo)
    r
end

cpyop(::Type{T}, f::Function, args...) where {T} = checknullconvert(T, cpyop(f, args...))

"""
    pyconvert(T, x) :: T

Convert Python object `x` to a `T`.
"""
pyconvert(::Type{T}, x) where {T} = checknullconvert(T, C.PyObject_From(x))
export pyconvert

"""
    pyis(x, y) :: Bool

Equivalent to `x is y` in Python.
"""
pyis(x::X, y::Y) where {X,Y} = begin
    if ispyreftype(X)
        xo = pyptr(x)
        isnull(xo) && pythrow()
    else
        xo = C.PyObject_From(x)
        isnull(xo) && pythrow()
        C.Py_DecRef(xo)
    end
    if ispyreftype(Y)
        yo = pyptr(y)
        isnull(yo) && pythrow()
    else
        yo = C.PyObject_From(y)
        isnull(yo) && pythrow()
        C.Py_DecRef(yo)
    end
    xo == yo
end
export pyis

"""
    pyhasattr(x, k) :: Bool

Equivalent to `hasattr(x, k)` in Python, returned as a `Bool`.
"""
pyhasattr(x, k) = checkm1(cpyop(C.PyObject_HasAttr, x, k)) != 0
pyhasattr(x, k::String) = checkm1(cpyop(xo -> C.PyObject_HasAttrString(xo, k), x)) != 0
pyhasattr(x, k::Symbol) = pyhasattr(x, string(k))
export pyhasattr

"""
    pygetattr([T=PyObject,] x, k) :: T

Equivalent to `x.k` or `getattr(x, k)` in Python.
"""
pygetattr(::Type{T}, x, k) where {T} = cpyop(T, C.PyObject_GetAttr, x, k)
pygetattr(::Type{T}, x, k::String) where {T} = cpyop(T, xo->C.PyObject_GetAttrString(xo, k), x)
pygetattr(::Type{T}, x, k::Symbol) where {T} = pygetattr(T, x, string(k))
pygetattr(x, k) = pygetattr(PyObject, x, k)
export pygetattr

"""
    pysetattr(x, k, v)

Equivalent to `x.k = v` or `setattr(x, k, v)` in Python, but returns `x`.
"""
pysetattr(x, k, v) = (checkm1(cpyop(C.PyObject_SetAttr, x, k, v)); x)
pysetattr(x, k::String, v) = (checkm1(cpyop((xo, vo) -> C.PyObject_SetAttrString(xo, k, vo), x, v)); x)
pysetattr(x, k::Symbol, v) = pysetattr(x, string(k), v)
export pysetattr

"""
    pydir([T=PyObject,] x) :: T

Equivalent to `dir(x)` in Python.
"""
pydir(::Type{T}, x) where {T} = cpyop(T, C.PyObject_Dir, x)
pydir(x) = pydir(PyObject, x)
export pydir

"""
    pycall([T=PyObject,] f, args...; kwargs...) :: T

Equivalent to `f(*args, **kwargs)` in Python.
"""
pycall(::Type{T}, f, args...; opts...) where {T} = cpyop(T, fo -> C.PyObject_CallArgs(fo, args, opts), f)
pycall(f, args...; opts...) = pycall(PyObject, f, args...; opts...)
export pycall

"""
    pyrepr([T=PyObject,] x) :: T

Equivalent to `repr(x)` in Python.
"""
pyrepr(::Type{T}, x) where {T} = cpyop(T, C.PyObject_Repr, x)
pyrepr(x) = pyrepr(PyObject, x)
export pyrepr

"""
    pystr([T=PyObject,] x) :: T

Equivalent to `str(x)` in Python.
"""
pystr(::Type{T}, x) where {T} = cpyop(T, C.PyObject_Str, x)
pystr(::Type{T}, x::Union{String, SubString{String}, Vector{Int8}, Vector{UInt8}}) where {T} = checknullconvert(T, C.PyUnicode_From(x))
pystr(x) = pystr(PyObject, x)
export pystr

"""
    pybytes([T=PyObject,] x) :: T

Equivalent to `str(x)` in Python.
"""
pybytes(::Type{T}, x) where {T} = cpyop(T, C.PyObject_Bytes, x)
pybytes(::Type{T}, x::Union{Vector{Int8}, Vector{UInt8}, String, SubString{String}}) where {T} = checknullconvert(T, C.PyBytes_From(x))
pybytes(x) = pybytes(PyObject, x)
export pybytes

"""
    pylen(x) :: Integer

Equivalent to `len(x)` in Python.
"""
pylen(x) = checkm1(cpyop(C.PyObject_Length, x))

"""
    pycontains(x, v) :: Bool

Equivalent to `v in x` in Python.
"""
pycontains(x, v) = checkm1(cpyop(C.PySequence_Contains, x, v)) != 0
export pycontains

"""
    pyin(v, x) :: Bool

Equivalent to `v in x` in Python.
"""
pyin(v, x) = pycontains(x, v)
export pyin

"""
    pygetitem([T=PyObject,] x, k) :: T

Equivalent to `x[k]` or `getitem(x, k)` in Python.
"""
pygetitem(::Type{T}, x, k) where {T} = cpyop(T, C.PyObject_GetItem, x, k)
pygetitem(x, k) = pygetitem(PyObject, x, k)
export pygetitem

"""
    pysetitem(x, k, v)

Equivalent to `x[k] = v` or `setitem(x, k, v)` in Python, but returns `x`.
"""
pysetitem(x, k, v) = (checkm1(cpyop(C.PyObject_SetItem, x, k, v)); x)
export pysetitem

"""
    pydelitem(x, k)

Equivalent to `del x[k]` or `delitem(x, k)` in Python, but returns x.
"""
pydelitem(x, k) = (checkm1(cpyop(C.PyObject_DelItem, x, k)); x)
export pydelitem

"""
    pynone([T=PyObject]) :: T

Equivalent to `None` in Python.
"""
pynone(::Type{T}) where {T} = (checkm1(C.PyObject_Convert(C.Py_None(), T)); takeresult(T))
pynone() = pynone(PyObject)
export pynone

"""
    pybool([T=PyObject,] ...) :: T

Equivalent to `bool(...)` in Python.
"""
pybool(::Type{T}, x::Bool) where {T} = checknullconvert(T, C.PyBool_From(x))
pybool(::Type{T}, args...; kwargs...) where {T} = checknullconvert(T, C.PyObject_CallArgs(C.PyBool_Type(), args, kwargs))
pybool(args...; kwargs...) = pybool(PyObject, args...; kwargs...)
export pybool

"""
    pyint([T=PyObject,] ...) :: T

Equivalent to `int(...)` in Python.
"""
pyint(::Type{T}, x::Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128,BigInt}) where {T} =
    checknullconvert(T, C.PyLong_From(x))
pyint(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Long, x)
pyint(::Type{T}, args...; kwargs...) where {T} = checknullconvert(T, C.PyObject_CallArgs(C.PyLong_Type(), args, kwargs))
pyint(args...; kwargs...) = pyint(PyObject, args...; kwargs...)
export pyint

"""
    pyfloat([T=PyObject,] ...) :: T

Equivalent to `float(...)` in Python.
"""
pyfloat(::Type{T}, x::Union{Float16,Float32,Float64}) where {T} = checknullconvert(T, C.PyFloat_From(x))
pyfloat(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Float, x)
pyfloat(::Type{T}, args...; kwargs...) where {T} = checknullconvert(T, C.PyObject_CallArgs(C.PyFloat_Type(), args, kwargs))
pyfloat(args...; kwargs...) = pyfloat(PyObject, args...; kwargs...)
export pyfloat

"""
    pyimport([T=PyObject,] name) :: T
    pyimport([T=PyObject,] name=>attr) :: T
    pyimport([T=PyObject,] name=>(attr,...)) :: Tuple{T,...}

Imports and returns the Python module `name`.

If additionally `attr` is given, the given attribute of the module is returned instead.
It may also be a tuple of attributes.

If several arguments are given, each one is imported and a tuple is returned.
"""
pyimport(::Type{T}, m) where {T} = cpyop(T, C.PyImport_Import, m)
pyimport(::Type{T}, m::String) where {T} = checknullconvert(T, C.PyImport_ImportModule(m))
pyimport(::Type{T}, m::Pair) where {T} = begin
    r = pyimport(PyObject, m[1])
    m[2] isa Tuple ? map(k->pygetattr(T, r, k), m[2]) : pygetattr(T, r, m[2])
end
pyimport(::Type{T}, m1, m2, ms...) where {T} = map(m->pyimport(T, m), (m1, m2, ms...))
pyimport(m1, ms...) = pyimport(PyObject, m1, ms...)
export pyimport

"""
    pytruth(x) :: Bool

The truthyness of `x`, equivalent to `bool(x)` or `not not x` in Python, or to `pybool(Bool, x)`.
"""
pytruth(x) = checkm1(cpyop(C.PyObject_IsTrue, x)) != 0
export pytruth

"""
    pyissubclass(x, y) :: Bool

Equivalent to `issubclass(x, y)` in Python.
"""
pyissubclass(x, y) = checkm1(cpyop(C.PyObject_IsSubclass, x, y)) != 0
export pyissubclass

"""
    pyisinstance(x, y) :: Bool

Equivalent to `isinstance(x, y)` in Python.
"""
pyisinstance(x, y) = checkm1(cpyop(C.PyObject_IsInstance, x, y)) != 0
export pyisinstance

"""
    pyhash(x) :: Integer

Equivalent to `hash(x)` in Python.
"""
pyhash(x) = checkm1(cpyop(C.PyObject_Hash, x))
export pyhash

"""
    pycompare([T=PyObject,] x, op, y) :: T

Equivalent to `x op y` in Python where `op` is one of: `=`, `≠`, `<`, `≤`, `>`, `≥`.
"""
pycompare(::Type{T}, x, ::typeof(==), y) where {T} = _pycompare(T, x, C.Py_EQ, y)
pycompare(::Type{T}, x, ::typeof(!=), y) where {T} = _pycompare(T, x, C.Py_NE, y)
pycompare(::Type{T}, x, ::typeof(< ), y) where {T} = _pycompare(T, x, C.Py_LT, y)
pycompare(::Type{T}, x, ::typeof(<=), y) where {T} = _pycompare(T, x, C.Py_LE, y)
pycompare(::Type{T}, x, ::typeof(> ), y) where {T} = _pycompare(T, x, C.Py_GT, y)
pycompare(::Type{T}, x, ::typeof(>=), y) where {T} = _pycompare(T, x, C.Py_GE, y)
pycompare(x, op, y) = pycompare(PyObject, x, op, y)
_pycompare(::Type{T}, x, op::Cint, y) where {T} = cpyop(T, (xo,yo)->C.PyObject_RichCompare(xo, yo, op), x, y)
_pycompare(::Type{Bool}, x, op::Cint, y) = checkm1(cpyop((xo,yo)->C.PyObject_RichCompareBool(xo, yo, op), x, y)) != 0
export pycompare

"""
    pyeq([T=PyObject,] x, y) :: T

Equivalent to `x == y` in Python.
"""
pyeq(::Type{T}, x, y) where {T} = pycompare(T, x, ==, y)
pyeq(x, y) = pycompare(x, ==, y)
export pyeq

"""
    pyne([T=PyObject,] x, y) :: T

Equivalent to `x != y` in Python.
"""
pyne(::Type{T}, x, y) where {T} = pycompare(T, x, !=, y)
pyne(x, y) = pycompare(x, !=, y)
export pyne

"""
    pyge([T=PyObject,] x, y) :: T

Equivalent to `x >= y` in Python.
"""
pyge(::Type{T}, x, y) where {T} = pycompare(T, x, >=, y)
pyge(x, y) = pycompare(x, >=, y)
export pyge

"""
    pygt([T=PyObject,] x, y) :: T

Equivalent to `x > y` in Python.
"""
pygt(::Type{T}, x, y) where {T} = pycompare(T, x, >, y)
pygt(x, y) = pycompare(x, >, y)
export pygt

"""
    pyle([T=PyObject,] x, y) :: T

Equivalent to `x <= y` in Python.
"""
pyle(::Type{T}, x, y) where {T} = pycompare(T, x, <=, y)
pyle(x, y) = pycompare(x, <=, y)
export pyle

"""
    pylt([T=PyObject,] x, y) :: T

Equivalent to `x < y` in Python.
"""
pylt(::Type{T}, x, y) where {T} = pycompare(T, x, <, y)
pylt(x, y) = pycompare(x, <, y)
export pylt

"""
    pyadd([T=PyObject,] x, y) :: T

Equivalent to `x + y` in Python.
"""
pyadd(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Add, x, y)
pyadd(x, y) = pyadd(PyObject, x, y)
export pyadd

"""
    pyiadd([T=typeof(x),] x, y) :: T

`x = pyiadd(x, y)` is equivalent to `x += y` in Python.
"""
pyiadd(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceAdd, x, y)
pyiadd(x, y) = pyiadd(typeof(x), x, y)
export pyiadd

"""
    pysub([T=PyObject,] x, y) :: T

Equivalent to `x - y` in Python.
"""
pysub(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Subtract, x, y)
pysub(x, y) = pysub(PyObject, x, y)
export pysub

"""
    pyisub([T=typeof(x),] x, y) :: T

`x = pyisub(x, y)` is equivalent to `x -= y` in Python.
"""
pyisub(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceSubtract, x, y)
pyisub(x, y) = pyisub(typeof(x), x, y)
export pyisub

"""
    pymul([T=PyObject,] x, y) :: T

Equivalent to `x * y` in Python.
"""
pymul(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Multiply, x, y)
pymul(x, y) = pymul(PyObject, x, y)
export pymul

"""
    pyimul([T=typeof(x),] x, y) :: T

`x = pyimul(x, y)` is equivalent to `x *= y` in Python.
"""
pyimul(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceMultiply, x, y)
pyimul(x, y) = pyimul(typeof(x), x, y)
export pyimul

"""
    pymatmul([T=PyObject,] x, y) :: T

Equivalent to `x @ y` in Python.
"""
pymatmul(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_MatrixMultiply, x, y)
pymatmul(x, y) = pymatmul(PyObject, x, y)
export pymatmul

"""
    pyimatmul([T=typeof(x),] x, y) :: T

`x = pyimatmul(x, y)` is equivalent to `x @= y` in Python.
"""
pyimatmul(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceMatrixMultiply, x, y)
pyimatmul(x, y) = pyimatmul(typeof(x), x, y)
export pyimatmul

"""
    pyfloordiv([T=PyObject,] x, y) :: T

Equivalent to `x // y` in Python.
"""
pyfloordiv(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_FloorDivide, x, y)
pyfloordiv(x, y) = pyfloordiv(PyObject, x, y)
export pyfloordiv

"""
    pyifloordiv([T=typeof(x),] x, y) :: T

`x = pyifloordiv(x, y)` is equivalent to `x //= y` in Python.
"""
pyifloordiv(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceFloorDivide, x, y)
pyifloordiv(x, y) = pyifloordiv(typeof(x), x, y)
export pyifloordiv

"""
    pytruediv([T=PyObject,] x, y) :: T

Equivalent to `x / y` in Python.
"""
pytruediv(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_TrueDivide, x, y)
pytruediv(x, y) = pytruediv(PyObject, x, y)
export pytruediv

"""
    pyitruediv([T=typeof(x),] x, y) :: T

`x = pyitruediv(x, y)` is equivalent to `x /= y` in Python.
"""
pyitruediv(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceTrueDivide, x, y)
pyitruediv(x, y) = pyitruediv(typeof(x), x, y)
export pyitruediv

"""
    pyrem([T=PyObject,] x, y) :: T

Equivalent to `x % y` in Python.
"""
pyrem(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Remainder, x, y)
pyrem(x, y) = pyrem(PyObject, x, y)
export pyrem

"""
    pyirem([T=typeof(x),] x, y) :: T

`x = pyirem(x, y)` is equivalent to `x %= y` in Python.
"""
pyirem(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceRemainder, x, y)
pyirem(x, y) = pyirem(typeof(x), x, y)
export pyirem

"""
    pydivmod([T=PyObject,] x, y) :: T

Equivalent to `divmod(x, y)` in Python.
"""
pydivmod(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_DivMod, x, y)
pydivmod(x, y) = pydivmod(PyObject, x, y)
export pydivmod

"""
    pylshift([T=PyObject,] x, y) :: T

Equivalent to `x << y` in Python.
"""
pylshift(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Lshift, x, y)
pylshift(x, y) = pylshift(PyObject, x, y)
export pylshift

"""
    pyilshift([T=typeof(x),] x, y) :: T

`x = pyilshift(x, y)` is equivalent to `x <<= y` in Python.
"""
pyilshift(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceLshift, x, y)
pyilshift(x, y) = pyilshift(typeof(x), x, y)
export pyilshift

"""
    pyrshift([T=PyObject,] x, y) :: T

Equivalent to `x >> y` in Python.
"""
pyrshift(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Rshift, x, y)
pyrshift(x, y) = pyrshift(PyObject, x, y)
export pyrshift

"""
    pyirshift([T=typeof(x),] x, y) :: T

`x = pyirshift(x, y)` is equivalent to `x >>= y` in Python.
"""
pyirshift(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceRshift, x, y)
pyirshift(x, y) = pyirshift(typeof(x), x, y)
export pyirshift

"""
    pyand([T=PyObject,] x, y) :: T

Equivalent to `x & y` in Python.
"""
pyand(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_And, x, y)
pyand(x, y) = pyand(PyObject, x, y)
export pyand

"""
    pyiand([T=typeof(x),] x, y) :: T

`x = pyiand(x, y)` is equivalent to `x &= y` in Python.
"""
pyiand(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceAnd, x, y)
pyiand(x, y) = pyiand(typeof(x), x, y)
export pyiand

"""
    pyxor([T=PyObject,] x, y) :: T

Equivalent to `x ^ y` in Python.
"""
pyxor(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Xor, x, y)
pyxor(x, y) = pyxor(PyObject, x, y)
export pyxor

"""
    pyixor([T=typeof(x),] x, y) :: T

`x = pyixor(x, y)` is equivalent to `x ^= y` in Python.
"""
pyixor(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceXor, x, y)
pyixor(x, y) = pyixor(typeof(x), x, y)
export pyixor

"""
    pyor([T=PyObject,] x, y) :: T

Equivalent to `x | y` in Python.
"""
pyor(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_Or, x, y)
pyor(x, y) = pyor(PyObject, x, y)
export pyor

"""
    pyior([T=typeof(x),] x, y) :: T

`x = pyior(x, y)` is equivalent to `x |= y` in Python.
"""
pyior(::Type{T}, x, y) where {T} = cpyop(T, C.PyNumber_InPlaceOr, x, y)
pyior(x, y) = pyior(typeof(x), x, y)
export pyior

"""
    pypow([T=PyObject,] x, y, [z]) :: T

Equivalent to `x**y` or `pow(x, y, z)` in Python.
"""
pypow(::Type{T}, x, y, z=C.PyObjectRef(C.Py_None())) where {T} = cpyop(T, C.PyNumber_Power, x, y, z)
pypow(x, y, z) = pypow(PyObject, x, y, z)
pypow(x, y) = pypow(PyObject, x, y)
export pypow

"""
    pyipow([T=typeof(x),] x, y, [z]) :: T

`x = pyipow(x, y)` is equivalent to `x **= y` in Python.
"""
pyipow(::Type{T}, x, y, z=C.PyObjectRef(C.Py_None())) where {T} = cpyop(T, C.PyNumber_InPlacePower, x, y, z)
pyipow(x, y, z) = pyipow(typeof(x), x, y, z)
pyipow(x, y) = pyipow(typeof(x), x, y)
export pyipow

"""
    pyneg([T=typeof(x),] x) :: T

Equivalent to `-x` in Python.
"""
pyneg(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Negative, x)
pyneg(x) = pyneg(typeof(x), x)
export pyneg

"""
    pypos([T=typeof(x),] x) :: T

Equivalent to `+x` in Python.
"""
pypos(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Positive, x)
pypos(x) = pypos(typeof(x), x)
export pypos

"""
    pyabs([T=typeof(x),] x) :: T

Equivalent to `abs(x)` in Python.
"""
pyabs(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Absolute, x)
pyabs(x) = pyabs(typeof(x), x)
export pyabs

"""
    pyinv([T=typeof(x),] x) :: T

Equivalent to `-x` in Python.
"""
pyinv(::Type{T}, x) where {T} = cpyop(T, C.PyNumber_Invert, x)
pyinv(x) = pyinv(typeof(x), x)
export pyinv

"""
    pyiter([T=PyObject] x) :: T

Equivalent to `iter(x)` in Python.
"""
pyiter(::Type{T}, x) where {T} = cpyop(T, C.PyObject_GetIter, x)
pyiter(x) = pyiter(PyObject, x)
export pyiter

"""
    pywith(f, o, d=nothing)

Equivalent to `with o as x: f(x)` in Python, where `x` is a `PyObject`.

On success, the value of `f(x)` is returned.
If an exception occurs but is suppressed then `d` is returned.
"""
function pywith(f, _o, d=nothing)
    o = PyObject(_o)
    t = pytype(o)
    exit = t.__exit__
    value = t.__enter__(o)
    exited = false
    try
        return f(value)
    catch err
        if err isa PyException
            exited = true
            if pytruth(exit(o, err.tref, err.vref, err.bref))
                rethrow()
            else
                return d
            end
        else
            rethrow()
        end
    finally
        exited || exit(o, pynone(), pynone(), pynone())
    end
end
export pywith

"""
    pytuple([T=PyObject,] [x]) :: T

Create a Python `tuple` from the elements of iterable `x`.

If `x` is a Python object, this is equivalent to `tuple(x)` in Python.
"""
pytuple(::Type{T}, x) where {T} = checknullconvert(T, ispyref(x) ? C.PyObject_CallNice(C.PyTuple_Type(), x) : C.PyTuple_FromIter(x))
pytuple(::Type{T}) where {T} = checknullconvert(T, C.PyTuple_New(0))
pytuple(x) = pytuple(PyObject, x)
pytuple() = pytuple(PyObject)
export pytuple

"""
    pylist([T=PyObject,] [x]) :: T

Create a Python `list` from the elements of iterable `x`.

If `x` is a Python object, this is equivalent to `list(x)` in Python.
"""
pylist(::Type{T}, x) where {T} = checknullconvert(T, ispyref(x) ? C.PyObject_CallNice(C.PyList_Type(), x) : C.PyList_FromIter(x))
pylist(::Type{T}) where {T} = checknullconvert(T, C.PyList_New(0))
pylist(x) = pylist(PyObject, x)
pylist() = pylist(PyObject)
export pylist

"""
    pycollist([T=PyObject,] x::AbstractArray) :: T

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of columns.
"""
pycollist(::Type{T}, x::AbstractArray) where {T} = ndims(x)==0 ? pyconvert(T, x[]) : pylist(T, pycollist(PyRef, y) for y in eachslice(x; dims=ndims(x)))
pycollist(x::AbstractArray) = pycollist(PyObject, x)
export pycollist

"""
    pyrowlist([T=PyObject,] x::AbstractArray) :: T

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of rows.
"""
pyrowlist(::Type{T}, x::AbstractArray) where {T} = ndims(x)==0 ? pyconvert(T, x[]) : pylist(T, pyrowlist(PyRef, y) for y in eachslice(x; dims=1))
pyrowlist(x::AbstractArray) = pyrowlist(PyObject, x)
export pyrowlist

"""
    pyset([T=PyObject,] [x]) :: T

Create a Python `set` from the elements of iterable `x`.

If `x` is a Python object, this is equivalent to `set(x)` in Python.
"""
pyset(::Type{T}, x) where {T} = checknullconvert(T, ispyref(x) ? C.PyObject_CallNice(C.PySet_Type(), x) : C.PySet_FromIter(x))
pyset(::Type{T}) where {T} = checknullconvert(T, C.PySet_New(C_NULL))
pyset(x) = pyset(PyObject, x)
pyset() = pyset(PyObject)
export pyset

"""
    pyfrozenset([T=PyObject,] [x]) :: T

Create a Python `frozenset` from the elements of iterable `x`.

If `x` is a Python object, this is equivalent to `frozenset(x)` in Python.
"""
pyfrozenset(::Type{T}, x) where {T} = checknullconvert(T, ispyref(x) ? C.PyObject_CallNice(C.PyFrozenSet_Type(), x) : C.PyFrozenSet_FromIter(x))
pyfrozenset(::Type{T}) where {T} = checknullconvert(T, C.PyFrozenSet_New(C_NULL))
pyfrozenset(x) = pyfrozenset(PyObject, x)
pyfrozenset() = pyfrozenset(PyObject)
export pyfrozenset

"""
    pydict([T=PyObject,] [x]) :: T
    pydict([T=PyObject;] key=value, ...)

Create a Python `dict` from the given key-value pairs in `x` or keyword arguments.

If `x` is a Python object, this is equivalent to `dict(x)` in Python.
"""
pydict(::Type{T}, x) where {T} = checknullconvert(T, ispyref(x) ? C.PyObject_CallNice(C.PyDict_Type(), x) : C.PyDict_FromPairs(x))
pydict(::Type{T}; opts...) where {T} = checknullconvert(T, isempty(opts) ? C.PyDict_New() : C.PyDict_FromStringPairs(opts))
pydict(x) = pydict(PyObject, x)
pydict(; opts...) = pydict(PyObject; opts...)
export pydict

"""
    pyslice([T=PyObject,] [start,] stop, [step]) :: T

Equivalent to `slice(start, stop, step)` in Python (or `start:stop:step` while indexing).
"""
pyslice(::Type{T}, x) where {T} = cpyop(T, x->C.PySlice_New(C_NULL, x, C_NULL), x)
pyslice(::Type{T}, x, y) where {T} = cpyop(T, (x,y)->C.PySlice_New(x, y, C_NULL), x, y)
pyslice(::Type{T}, x, y, z) where {T} = cpyop(T, C.PySlice_New, x, y, z)
pyslice(x) = pyslice(PyObject, x)
pyslice(x, y) = pyslice(PyObject, x, y)
pyslice(x, y, z) = pyslice(PyObject, x, y, z)
export pyslice

"""
    pyellipsis([T=PyObject]) :: T

Equivalent to `Ellipsis` in Python (or `...` while indexing).
"""
pyellipsis(::Type{T}) where {T} = checknullconvert(T, C.PyEllipsis_New())
pyellipsis() = pyellipsis(PyObject)
export pyellipsis

"""
    pynotimplemented([T=PyObject]) :: T

Equivalent to `NotImplemented` in Python.
"""
pynotimplemented(::Type{T}) where {T} = checknullconvert(T, C.PyNotImplemented_New())
pynotimplemented() = pynotimplemented(PyObject)
export pynotimplemented

"""
    pymethod([T=PyObject,] x) :: T

Convert `x` to a Python instance method.
"""
pymethod(::Type{T}, x) where {T} = cpyop(T, C.PyInstanceMethod_New, x)
pymethod(x) = pymethod(PyObject, x)
export pymethod

"""
    pytype([T=PyObject,] x) :: T

Equivalent to `type(x)` in Python.
"""
pytype(::Type{T}, x) where {T} = cpyop(T, o -> (t=C.Py_Type(o); C.Py_IncRef(t); t), x)
pytype(x) = pytype(PyObject, x)
export pytype

"""
    pytype([T=PyObject,] name, bases, dict) :: T

Equivalent to `type(name, bases, dict)` in Python.
"""
pytype(::Type{T}, name, bases, dict) where {T} = @pyv `type($name, $(pytuple(bases)), $(dict isa NamedTuple ? pydict(;dict...) : pydict(dict)))`::T
pytype(name, bases, dict) = pytype(PyObject, name, bases, dict)
export pytype

### MULTIMEDIA DISPLAY

const _py_mimes = [
    (MIME"text/html", "_repr_html_"), (MIME"text/markdown", "_repr_markdown_"),
    (MIME"text/json", "_repr_json_"), (MIME"application/javascript", "_repr_javascript_"),
    (MIME"application/pdf", "_repr_pdf_"), (MIME"image/jpeg", "_repr_jpeg_"),
    (MIME"image/png", "_repr_png_"), (MIME"image/svg+xml", "_repr_svg_"),
    (MIME"text/latex", "_repr_latex_")
]
const _py_mimetype = Union{map(first, _py_mimes)...}

for (mime, method) in _py_mimes
    T = istextmime(mime()) ? String : Vector{UInt8}
    @eval begin
        _py_mime_show(io::IO, mime::$mime, o) = begin
            try
                x = pycall(PyRef, pygetattr(PyRef, o, $method))
                pyis(x, pynone(PyRef)) || return write(io, pyconvert($T, x))
            catch
            end
            throw(MethodError(_py_mime_show, (io, mime, o)))
        end
        _py_mime_showable(::$mime, o) = begin
            try
                x = pycall(PyRef, pygetattr(PyRef, o, $method))
                if pyis(x, pynone(PyRef))
                    false
                else
                    pyconvert($T, x)
                    true
                end
            catch
                false
            end
        end
    end
end

### IO

"""
    pytextio([T=PyObject], io::IO) :: T

Convert `io` to a Python text IO stream, specifically a `julia.TextIOValue`.
"""
pytextio(::Type{T}, io::IO) where {T} = checknullconvert(T, C.PyJuliaTextIOValue_New(io))
pytextio(io::IO) = pytextio(PyObject, io)
export pytextio

"""
    pybufferedio([T=PyObject], io::IO) :: T

Convert `io` to a Python buffered byte IO stream, specifically a `julia.BufferedIOValue`.
"""
pybufferedio(::Type{T}, io::IO) where {T} = checknullconvert(T, C.PyJuliaBufferedIOValue_New(io))
pybufferedio(io::IO) = pybufferedio(PyObject, io)
export pybufferedio

"""
    pyrawio([T=PyObject], io::IO) :: T

Convert `io` to a Python raw (unbuffered byte) IO stream, specifically a `julia.RawIOValue`.
"""
pyrawio(::Type{T}, io::IO) where {T} = checknullconvert(T, C.PyJuliaRawIOValue_New(io))
pyrawio(io::IO) = pyrawio(PyObject, io)
export pyrawio
