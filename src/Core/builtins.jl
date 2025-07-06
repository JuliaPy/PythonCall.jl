### object interface

"""
    pyis(x, y)

True if `x` and `y` are the same Python object. Equivalent to `x is y` in Python.
"""
pyis(x, y) = @autopy x y getptr(x_) == getptr(y_)
export pyis

pyisnot(x, y) = !pyis(x, y)

"""
    pyrepr(x)

Equivalent to `repr(x)` in Python.
"""
pyrepr(x) = pynew(errcheck(@autopy x C.PyObject_Repr(x_)))
pyrepr(::Type{String}, x) = (s = pyrepr(x); ans = pystr_asstring(s); pydel!(s); ans)
export pyrepr

"""
    pyascii(x)

Equivalent to `ascii(x)` in Python.
"""
pyascii(x) = pynew(errcheck(@autopy x C.PyObject_ASCII(x_)))
pyascii(::Type{String}, x) = (s = pyascii(x); ans = pystr_asstring(s); pydel!(s); ans)
export pyascii

"""
    pyhasattr(x, k)

Equivalent to `hasattr(x, k)` in Python.

Tests if `getattr(x, k)` raises an `AttributeError`.
"""
function pyhasattr(x, k)
    ptr = @autopy x k C.PyObject_GetAttr(x_, k_)
    if iserrset(ptr)
        if errmatches(pybuiltins.AttributeError)
            errclear()
            return false
        else
            pythrow()
        end
    else
        decref(ptr)
        return true
    end
end
# pyhasattr(x, k) = errcheck(@autopy x k C.PyObject_HasAttr(x_, k_)) == 1
export pyhasattr

"""
    pygetattr(x, k, [d])

Equivalent to `getattr(x, k)` or `x.k` in Python.

If `d` is specified, it is returned if the attribute does not exist.
"""
pygetattr(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetAttr(x_, k_)))
function pygetattr(x, k, d)
    ptr = @autopy x k C.PyObject_GetAttr(x_, k_)
    if iserrset(ptr)
        if errmatches(pybuiltins.AttributeError)
            errclear()
            return d
        else
            pythrow()
        end
    else
        return pynew(ptr)
    end
end
export pygetattr

"""
    pysetattr(x, k, v)

Equivalent to `setattr(x, k, v)` or `x.k = v` in Python.
"""
pysetattr(x, k, v) = (
    errcheck(@autopy x k v C.PyObject_SetAttr(x_, k_, v_)); nothing
)
export pysetattr

"""
    pydelattr(x, k)

Equivalent to `delattr(x, k)` or `del x.k` in Python.
"""
pydelattr(x, k) =
    (errcheck(@autopy x k C.PyObject_SetAttr(x_, k_, C.PyNULL)); nothing)
export pydelattr

"""
    pyissubclass(s, t)

Test if `s` is a subclass of `t`. Equivalent to `issubclass(s, t)` in Python.
"""
pyissubclass(s, t) =
    errcheck(@autopy s t C.PyObject_IsSubclass(s_, t_)) == 1
export pyissubclass

"""
    pyisinstance(x, t)

Test if `x` is of type `t`. Equivalent to `isinstance(x, t)` in Python.
"""
pyisinstance(x, t) =
    errcheck(@autopy x t C.PyObject_IsInstance(x_, t_)) == 1
export pyisinstance

"""
    pyhash(x)

Equivalent to `hash(x)` in Python, converted to an `Integer`.
"""
pyhash(x) = errcheck(@autopy x C.PyObject_Hash(x_))
export pyhash

"""
    pytruth(x)

The truthyness of `x`. Equivalent to `bool(x)` in Python, converted to a `Bool`.
"""
pytruth(x) = errcheck(@autopy x C.PyObject_IsTrue(x_)) == 1
export pytruth

"""
    pynot(x)

The falsyness of `x`. Equivalent to `not x` in Python, converted to a `Bool`.
"""
pynot(x) = errcheck(@autopy x C.PyObject_Not(x_)) == 1
export pynot

"""
    pylen(x)

The length of `x`. Equivalent to `len(x)` in Python, converted to an `Integer`.
"""
pylen(x) = errcheck(@autopy x C.PyObject_Length(x_))
export pylen

"""
    pyhasitem(x, k)

Test if `pygetitem(x, k)` raises a `KeyError` or `AttributeError`.
"""
function pyhasitem(x, k)
    ptr = @autopy x k C.PyObject_GetItem(x_, k_)
    if iserrset(ptr)
        if errmatches(pybuiltins.KeyError) || errmatches(pybuiltins.IndexError)
            errclear()
            return false
        else
            pythrow()
        end
    else
        decref(ptr)
        return true
    end
end
export pyhasitem

"""
    pygetitem(x, k, [d])

Equivalent `x[k]` in Python.

If `d` is specified, it is returned if the item does not exist (i.e. if `x[k]` raises a
`KeyError` or `IndexError`).
"""
pygetitem(x, k) = pynew(errcheck(@autopy x k C.PyObject_GetItem(x_, k_)))
function pygetitem(x, k, d)
    ptr = @autopy x k C.PyObject_GetItem(x_, k_)
    if iserrset(ptr)
        if errmatches(pybuiltins.KeyError) || errmatches(pybuiltins.IndexError)
            errclear()
            return d
        else
            pythrow()
        end
    else
        return pynew(ptr)
    end
end
export pygetitem

"""
    pysetitem(x, k, v)

Equivalent to `setitem(x, k, v)` or `x[k] = v` in Python.
"""
pysetitem(x, k, v) = (
    errcheck(@autopy x k v C.PyObject_SetItem(x_, k_, v_)); nothing
)
export pysetitem

"""
    pydelitem(x, k)

Equivalent to `delitem(x, k)` or `del x[k]` in Python.
"""
pydelitem(x, k) =
    (errcheck(@autopy x k C.PyObject_DelItem(x_, k_)); nothing)
export pydelitem

"""
    pydir(x)

Equivalent to `dir(x)` in Python.
"""
pydir(x) = pynew(errcheck(@autopy x C.PyObject_Dir(x_)))
export pydir

pycallargs(f) = pynew(errcheck(@autopy f C.PyObject_CallObject(f_, C.PyNULL)))
pycallargs(f, args) =
    pynew(errcheck(@autopy f args C.PyObject_CallObject(f_, args_)))
pycallargs(f, args, kwargs) = pynew(
    errcheck(
        @autopy f args kwargs C.PyObject_Call(f_, args_, kwargs_)
    ),
)

"""
    pycall(f, args...; kwargs...)

Call the Python object `f` with the given arguments.
"""
pycall(f, args...; kwargs...) =
    if !isempty(kwargs)
        args_ = pytuple_fromiter(args)
        kwargs_ = pystrdict_fromiter(kwargs)
        ans = pycallargs(f, args_, kwargs_)
        pydel!(args_)
        pydel!(kwargs_)
        ans
    elseif !isempty(args)
        args_ = pytuple_fromiter(args)
        ans = pycallargs(f, args_)
        pydel!(args_)
        ans
    else
        pycallargs(f)
    end
export pycall

"""
    pyeq(x, y)
    pyeq(Bool, x, y)

Equivalent to `x == y` in Python. The second form converts to `Bool`.
"""
pyeq(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_EQ)))

"""
    pyne(x, y)
    pyne(Bool, x, y)

Equivalent to `x != y` in Python. The second form converts to `Bool`.
"""
pyne(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_NE)))

"""
    pyle(x, y)
    pyle(Bool, x, y)

Equivalent to `x <= y` in Python. The second form converts to `Bool`.
"""
pyle(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_LE)))

"""
    pylt(x, y)
    pylt(Bool, x, y)

Equivalent to `x < y` in Python. The second form converts to `Bool`.
"""
pylt(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_LT)))

"""
    pyge(x, y)
    pyge(Bool, x, y)

Equivalent to `x >= y` in Python. The second form converts to `Bool`.
"""
pyge(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_GE)))

"""
    pygt(x, y)
    pygt(Bool, x, y)

Equivalent to `x > y` in Python. The second form converts to `Bool`.
"""
pygt(x, y) =
    pynew(errcheck(@autopy x y C.PyObject_RichCompare(x_, y_, C.Py_GT)))
pyeq(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_EQ)) == 1
pyne(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_NE)) == 1
pyle(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_LE)) == 1
pylt(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_LT)) == 1
pyge(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_GE)) == 1
pygt(::Type{Bool}, x, y) =
    errcheck(@autopy x y C.PyObject_RichCompareBool(x_, y_, C.Py_GT)) == 1
export pyeq, pyne, pyle, pylt, pyge, pygt

"""
    pycontains(x, v)

Equivalent to `v in x` in Python.
"""
pycontains(x, v) = errcheck(@autopy x v C.PySequence_Contains(x_, v_)) == 1
export pycontains

"""
    pyin(v, x)

Equivalent to `v in x` in Python.
"""
pyin(v, x) = pycontains(x, v)
export pyin

pynotin(v, x) = !pyin(v, x)

### number interface

# unary
"""
    pyneg(x)

Equivalent to `-x` in Python.
"""
pyneg(x) = pynew(errcheck(@autopy x C.PyNumber_Negative(x_)))
"""
    pypos(x)

Equivalent to `+x` in Python.
"""
pypos(x) = pynew(errcheck(@autopy x C.PyNumber_Positive(x_)))
"""
    pyabs(x)

Equivalent to `abs(x)` in Python.
"""
pyabs(x) = pynew(errcheck(@autopy x C.PyNumber_Absolute(x_)))
"""
    pyinv(x)

Equivalent to `~x` in Python.
"""
pyinv(x) = pynew(errcheck(@autopy x C.PyNumber_Invert(x_)))
"""
    pyindex(x)

Convert `x` losslessly to an `int`.
"""
pyindex(x) = pynew(errcheck(@autopy x C.PyNumber_Index(x_)))
export pyneg, pypos, pyabs, pyinv, pyindex

# binary
"""
    pyadd(x, y)

Equivalent to `x + y` in Python.
"""
pyadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Add(x_, y_)))
"""
    pysub(x, y)

Equivalent to `x - y` in Python.
"""
pysub(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Subtract(x_, y_)))
"""
    pymul(x, y)

Equivalent to `x * y` in Python.
"""
pymul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Multiply(x_, y_)))
"""
    pymatmul(x, y)

Equivalent to `x @ y` in Python.
"""
pymatmul(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_MatrixMultiply(x_, y_)))
"""
    pyfloordiv(x, y)

Equivalent to `x // y` in Python.
"""
pyfloordiv(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_FloorDivide(x_, y_)))
"""
    pytruediv(x, y)

Equivalent to `x / y` in Python.
"""
pytruediv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_TrueDivide(x_, y_)))
"""
    pymod(x, y)

Equivalent to `x % y` in Python.
"""
pymod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Remainder(x_, y_)))
"""
    pydivmod(x, y)

Equivalent to `divmod(x, y)` in Python.
"""
pydivmod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Divmod(x_, y_)))
"""
    pylshift(x, y)

Equivalent to `x << y` in Python.
"""
pylshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Lshift(x_, y_)))
"""
    pyrshift(x, y)

Equivalent to `x >> y` in Python.
"""
pyrshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Rshift(x_, y_)))
"""
    pyand(x, y)

Equivalent to `x & y` in Python.
"""
pyand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_And(x_, y_)))
"""
    pyxor(x, y)

Equivalent to `x ^ y` in Python.
"""
pyxor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Xor(x_, y_)))
"""
    pyor(x, y)

Equivalent to `x | y` in Python.
"""
pyor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Or(x_, y_)))
export pyadd,
    pysub,
    pymul,
    pymatmul,
    pyfloordiv,
    pytruediv,
    pymod,
    pydivmod,
    pylshift,
    pyrshift,
    pyand,
    pyxor,
    pyor

# binary in-place
"""
    pyiadd(x, y)

In-place add. `x = pyiadd(x, y)` is equivalent to `x += y` in Python.
"""
pyiadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAdd(x_, y_)))
"""
    pyisub(x, y)

In-place subtract. `x = pyisub(x, y)` is equivalent to `x -= y` in Python.
"""
pyisub(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceSubtract(x_, y_)))
"""
    pyimul(x, y)

In-place multiply. `x = pyimul(x, y)` is equivalent to `x *= y` in Python.
"""
pyimul(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceMultiply(x_, y_)))
"""
    pyimatmul(x, y)

In-place matrix multiply. `x = pyimatmul(x, y)` is equivalent to `x @= y` in Python.
"""
pyimatmul(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceMatrixMultiply(x_, y_)))
"""
    pyifloordiv(x, y)

In-place floor divide. `x = pyifloordiv(x, y)` is equivalent to `x //= y` in Python.
"""
pyifloordiv(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceFloorDivide(x_, y_)))
"""
    pyitruediv(x, y)

In-place true division. `x = pyitruediv(x, y)` is equivalent to `x /= y` in Python.
"""
pyitruediv(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceTrueDivide(x_, y_)))
"""
    pyimod(x, y)

In-place subtraction. `x = pyimod(x, y)` is equivalent to `x %= y` in Python.
"""
pyimod(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceRemainder(x_, y_)))
"""
    pyilshift(x, y)

In-place left shift. `x = pyilshift(x, y)` is equivalent to `x <<= y` in Python.
"""
pyilshift(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceLshift(x_, y_)))
"""
    pyirshift(x, y)

In-place right shift. `x = pyirshift(x, y)` is equivalent to `x >>= y` in Python.
"""
pyirshift(x, y) =
    pynew(errcheck(@autopy x y C.PyNumber_InPlaceRshift(x_, y_)))
"""
    pyiand(x, y)

In-place and. `x = pyiand(x, y)` is equivalent to `x &= y` in Python.
"""
pyiand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAnd(x_, y_)))
"""
    pyixor(x, y)

In-place xor. `x = pyixor(x, y)` is equivalent to `x ^= y` in Python.
"""
pyixor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceXor(x_, y_)))
"""
    pyior(x, y)

In-place or. `x = pyior(x, y)` is equivalent to `x |= y` in Python.
"""
pyior(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceOr(x_, y_)))
export pyiadd,
    pyisub,
    pyimul,
    pyimatmul,
    pyifloordiv,
    pyitruediv,
    pyimod,
    pyilshift,
    pyirshift,
    pyiand,
    pyixor,
    pyior

# power
"""
    pypow(x, y, z=None)

Equivalent to `x ** y` or `pow(x, y, z)` in Python.
"""
pypow(x, y, z = pybuiltins.None) =
    pynew(errcheck(@autopy x y z C.PyNumber_Power(x_, y_, z_)))
"""
    pyipow(x, y, z=None)

In-place power. `x = pyipow(x, y)` is equivalent to `x **= y` in Python.
"""
pyipow(x, y, z = pybuiltins.None) = pynew(
    errcheck(@autopy x y z C.PyNumber_InPlacePower(x_, y_, z_)),
)
export pypow, pyipow

### iter

"""
    pyiter(x)

Equivalent to `iter(x)` in Python.
"""
pyiter(x) = pynew(errcheck(@autopy x C.PyObject_GetIter(x_)))
export pyiter

"""
    pynext(x)

Equivalent to `next(x)` in Python.
"""
pynext(x) = pybuiltins.next(x)
export pynext

"""
    unsafe_pynext(x)

Return the next item in the iterator `x`. When there are no more items, return NULL.
"""
unsafe_pynext(x::Py) = Base.GC.@preserve x pynew(errcheck_ambig(C.PyIter_Next(x)))

### None

pyisnone(x) = pyis(x, pybuiltins.None)

### bool

"""
    pybool(x)

Convert `x` to a Python `bool`.
"""
pybool(x::Bool = false) = pynew(x ? pybuiltins.True : pybuiltins.False)
pybool(x::Number) = pybool(!iszero(x))
pybool(x) = pybuiltins.bool(x)
export pybool

pyisTrue(x) = pyis(x, pybuiltins.True)
pyisFalse(x) = pyis(x, pybuiltins.False)
pyisbool(x) = pyisTrue(x) || pyisFalse(x)

function pybool_asbool(x)
    @autopy x if pyisTrue(x_)
        true
    elseif pyisFalse(x_)
        false
    else
        error("not a bool")
    end
end

### str

pystr_fromUTF8(x::Ptr, n::Integer) = pynew(errcheck(C.PyUnicode_DecodeUTF8(x, n, C_NULL)))
pystr_fromUTF8(x) = pystr_fromUTF8(pointer(x), sizeof(x))

"""
    pystr(x)

Convert `x` to a Python `str`.
"""
pystr(x) = pynew(errcheck(@autopy x C.PyObject_Str(x_)))
pystr(x::String) = pystr_fromUTF8(x)
pystr(x::SubString{String}) = pystr_fromUTF8(x)
pystr(x::Char) = pystr(string(x))
pystr(::Type{String}, x) = (s = pystr(x); ans = pystr_asstring(s); pydel!(s); ans)
export pystr

pystr_asUTF8bytes(x::Py) = pynew(errcheck(C.PyUnicode_AsUTF8String(x)))
pystr_asUTF8vector(x::Py) =
    (b = pystr_asUTF8bytes(x); ans = pybytes_asvector(b); pydel!(b); ans)
pystr_asstring(x::Py) =
    (b = pystr_asUTF8bytes(x); ans = pybytes_asUTF8string(b); pydel!(b); ans)

function pystr_intern!(x::Py)
    ptr = Ref(getptr(x))
    C.PyUnicode_InternInPlace(ptr)
    setptr!(x, ptr[])
end

pyisstr(x) = pytypecheckfast(x, C.Py_TPFLAGS_UNICODE_SUBCLASS)

### bytes

pybytes_fromdata(x::Ptr, n::Integer) = pynew(errcheck(C.PyBytes_FromStringAndSize(x, n)))
pybytes_fromdata(x) = pybytes_fromdata(pointer(x), sizeof(x))

"""
    pybytes(x)

Convert `x` to a Python `bytes`.
"""
pybytes(x) = pynew(errcheck(@autopy x C.PyObject_Bytes(x_)))
pybytes(x::Vector{UInt8}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8,String}) = pybytes_fromdata(x)
pybytes(x::Base.CodeUnits{UInt8,SubString{String}}) = pybytes_fromdata(x)
pybytes(::Type{T}, x) where {Vector{UInt8} <: T <: Vector} =
    (b = pybytes(x); ans = pybytes_asvector(b); pydel!(b); ans)
pybytes(::Type{T}, x) where {Base.CodeUnits{UInt8,String} <: T <: Base.CodeUnits} =
    (b = pybytes(x); ans = Base.CodeUnits(pybytes_asUTF8string(b)); pydel!(b); ans)
export pybytes

pyisbytes(x) = pytypecheckfast(x, C.Py_TPFLAGS_BYTES_SUBCLASS)

function pybytes_asdata(x::Py)
    ptr = Ref(Ptr{Cchar}(0))
    len = Ref(C.Py_ssize_t(0))
    errcheck(C.PyBytes_AsStringAndSize(x, ptr, len))
    ptr[], len[]
end

function pybytes_asvector(x::Py)
    ptr, len = pybytes_asdata(x)
    unsafe_wrap(Array, Ptr{UInt8}(ptr), len)
end

function pybytes_asUTF8string(x::Py)
    ptr, len = pybytes_asdata(x)
    unsafe_string(Ptr{UInt8}(ptr), len)
end

### int

pyint_fallback(
    x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt},
) = pynew(errcheck(C.PyLong_FromString(string(x, base = 32), C_NULL, 32)))
pyint_fallback(x::Integer) = pyint_fallback(BigInt(x))

"""
    pyint(x=0)

Convert `x` to a Python `int`.
"""
function pyint(x::Integer = 0)
    y = mod(x, Clonglong)
    if x == y
        pynew(errcheck(C.PyLong_FromLongLong(y)))
    else
        pyint_fallback(x)
    end
end
function pyint(x::Unsigned)
    y = mod(x, Culonglong)
    if x == y
        pynew(errcheck(C.PyLong_FromUnsignedLongLong(y)))
    else
        pyint_fallback(x)
    end
end
pyint(x) = @autopy x pynew(errcheck(C.PyNumber_Long(x_)))
export pyint

pyisint(x) = pytypecheckfast(x, C.Py_TPFLAGS_LONG_SUBCLASS)

### float

"""
    pyfloat(x=0.0)

Convert `x` to a Python `float`.
"""
pyfloat(x::Real = 0.0) = pynew(errcheck(C.PyFloat_FromDouble(x)))
pyfloat(x) = @autopy x pynew(errcheck(C.PyNumber_Float(x_)))
export pyfloat

pyisfloat(x) = pytypecheck(x, pybuiltins.float)

pyfloat_asdouble(x) = errcheck_ambig(@autopy x C.PyFloat_AsDouble(x_))

### complex

"""
    pycomplex(x=0.0)
    pycomplex(re, im)

Convert `x` to a Python `complex`, or create one from given real and imaginary parts.
"""
pycomplex(x::Real = 0.0, y::Real = 0.0) = pynew(errcheck(C.PyComplex_FromDoubles(x, y)))
pycomplex(x::Complex) = pycomplex(real(x), imag(x))
pycomplex(x) = pybuiltins.complex(x)
pycomplex(x, y) = pybuiltins.complex(x, y)
export pycomplex

pyiscomplex(x) = pytypecheck(x, pybuiltins.complex)

function pycomplex_ascomplex(x)
    c = @autopy x C.PyComplex_AsCComplex(x_)
    c.real == -1 && c.imag == 0 && errcheck()
    return Complex(c.real, c.imag)
end

### type

"""
    pytype(x)

The Python `type` of `x`.
"""
pytype(x) = pynew(errcheck(@autopy x C.PyObject_Type(x_)))
export pytype

"""
    pytype(name, bases, dict)

Create a new type. Equivalent to `type(name, bases, dict)` in Python.

If `bases` is not a Python object, it is converted to one using `pytuple`.

The `dict` may either by a Python object or a Julia iterable. In the latter case, each item
may either be a `name => value` pair or a Python object with a `__name__` attribute.

In order to use a Julia `Function` as an instance method, it must be wrapped into a Python
function with [`pyfunc`](@ref PythonCall.pyfunc). Similarly, see also [`pyclassmethod`](@ref PythonCall.pyclassmethod),
[`pystaticmethod`](@ref PythonCall.pystaticmethod) or [`pyproperty`](@ref PythonCall.pyproperty). In all these cases, the arguments passed
to the function always have type `Py`. See the example below.

# Example

```julia
Foo = pytype("Foo", (), [
    "__module__" => "__main__",

    pyfunc(
        name = "__init__",
        doc = \"\"\"
        Specify x and y to store in the Foo.

        If omitted, y defaults to None.
        \"\"\",
        function (self, x, y = nothing)
            self.x = x
            self.y = y
            return
        end,
    ),

    pyfunc(
        name = "__repr__",
        self -> "Foo(\$(self.x), \$(self.y))",
    ),

    pyclassmethod(
        name = "frompair",
        doc = "Construct a Foo from a tuple of length two.",
        (cls, xy) -> cls(xy...),
    ),

    pystaticmethod(
        name = "hello",
        doc = "Prints a friendly greeting.",
        (name) -> println("Hello, \$name"),
    ),

    "xy" => pyproperty(
        doc = "A tuple of x and y.",
        get = (self) -> (self.x, self.y),
        set = function (self, xy)
            (x, y) = xy
            self.x = x
            self.y = y
            nothing
        end,
    ),
])
```
"""
function pytype(name, bases, dict)
    bases2 = ispy(bases) ? bases : pytuple(bases)
    dict2 =
        ispy(dict) ? dict :
        pydict(ispy(item) ? (pygetattr(item, "__name__") => item) : item for item in dict)
    pybuiltins.type(name, bases2, dict2)
end

pyistype(x) = pytypecheckfast(x, C.Py_TPFLAGS_TYPE_SUBCLASS)

pytypecheck(x, t) = (@autopy x t C.Py_TypeCheck(x_, t_)) == 1
pytypecheckfast(x, f) = (@autopy x C.Py_TypeCheckFast(x_, f)) == 1

### slice

"""
    pyslice([start], stop, [step])

Construct a Python `slice`. Unspecified arguments default to `None`.
"""
pyslice(x, y, z = pybuiltins.None) =
    pynew(errcheck(@autopy x y z C.PySlice_New(x_, y_, z_)))
pyslice(y) = pyslice(pybuiltins.None, y, pybuiltins.None)
export pyslice

pyisslice(x) = pytypecheck(x, pybuiltins.slice)

### range

"""
    pyrange([[start], [stop]], [step])

Construct a Python `range`. Unspecified arguments default to `None`.
"""
pyrange(x, y, z) = pybuiltins.range(x, y, z)
pyrange(x, y) = pybuiltins.range(x, y)
pyrange(y) = pybuiltins.range(y)
export pyrange

pyrange_fromrange(x::AbstractRange) = pyrange(first(x), last(x) + sign(step(x)), step(x))

pyisrange(x) = pytypecheck(x, pybuiltins.range)

### tuple

pynulltuple(len) = pynew(errcheck(C.PyTuple_New(len)))

function pytuple_setitem(xs::Py, i, x)
    errcheck(C.PyTuple_SetItem(xs, i, incref(Py(x))))
    return xs
end

function pytuple_getitem(xs::Py, i)
    Base.GC.@preserve xs pynew(incref(errcheck(C.PyTuple_GetItem(xs, i))))
end

function pytuple_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known, e.g. Tuple, Pair, Vector
        ans = pynulltuple(length(xs))
        for (i, x) in enumerate(xs)
            pytuple_setitem(ans, i - 1, x)
        end
        return ans
    else
        # length unknown
        xs_ = pylist_fromiter(xs)
        ans = pylist_astuple(xs_)
        pydel!(xs_)
        return ans
    end
end

@generated function pytuple_fromiter(xs::Tuple)
    n = length(xs.parameters)
    code = []
    push!(code, :(ans = pynulltuple($n)))
    for i = 1:n
        push!(code, :(pytuple_setitem(ans, $(i - 1), xs[$i])))
    end
    push!(code, :(return ans))
    return Expr(:block, code...)
end

"""
    pytuple(x=())

Convert `x` to a Python `tuple`.

If `x` is a Python object, this is equivalent to `tuple(x)` in Python.
Otherwise `x` must be iterable.
"""
pytuple() = pynulltuple(0)
pytuple(x) = ispy(x) ? pybuiltins.tuple(x) : pytuple_fromiter(x)
export pytuple

pyistuple(x) = pytypecheckfast(x, C.Py_TPFLAGS_TUPLE_SUBCLASS)

### list

pynulllist(len) = pynew(errcheck(C.PyList_New(len)))

function pylist_setitem(xs::Py, i, x)
    errcheck(C.PyList_SetItem(xs, i, incref(Py(x))))
    return xs
end

pylist_append(xs::Py, x) = errcheck(@autopy x C.PyList_Append(xs, x_))

pylist_astuple(x) = pynew(errcheck(@autopy x C.PyList_AsTuple(x_)))

function pylist_fromiter(xs)
    sz = Base.IteratorSize(typeof(xs))
    if sz isa Base.HasLength || sz isa Base.HasShape
        # length known
        ans = pynulllist(length(xs))
        for (i, x) in enumerate(xs)
            pylist_setitem(ans, i - 1, x)
        end
        return ans
    else
        # length unknown
        ans = pynulllist(0)
        for x in xs
            pylist_append(ans, x)
        end
        return ans
    end
end

"""
    pylist(x=())

Convert `x` to a Python `list`.

If `x` is a Python object, this is equivalent to `list(x)` in Python.
Otherwise `x` must be iterable.
"""
pylist() = pynulllist(0)
pylist(x) = ispy(x) ? pybuiltins.list(x) : pylist_fromiter(x)
export pylist

"""
    pycollist(x::AbstractArray)

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of columns.
"""
function pycollist(x::AbstractArray{T,N}) where {T,N}
    N == 0 && return pynew(Py(x[]))
    d = N
    ax = axes(x, d)
    ans = pynulllist(length(ax))
    for (i, j) in enumerate(ax)
        y = pycollist(selectdim(x, d, j))
        pylist_setitem(ans, i - 1, y)
        pydel!(y)
    end
    return ans
end
export pycollist

"""
    pyrowlist(x::AbstractArray)

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of rows.
"""
function pyrowlist(x::AbstractArray{T,N}) where {T,N}
    ndims(x) == 0 && return pynew(Py(x[]))
    d = 1
    ax = axes(x, d)
    ans = pynulllist(length(ax))
    for (i, j) in enumerate(ax)
        y = pyrowlist(selectdim(x, d, j))
        pylist_setitem(ans, i - 1, y)
        pydel!(y)
    end
    return ans
end
export pyrowlist

### set

pyset_add(set::Py, x) = (errcheck(@autopy x C.PySet_Add(set, x_)); set)

function pyset_update_fromiter(set::Py, xs)
    for x in xs
        pyset_add(set, x)
    end
    return set
end
pyset_fromiter(xs) = pyset_update_fromiter(pyset(), xs)
pyfrozenset_fromiter(xs) = pyset_update_fromiter(pyfrozenset(), xs)

"""
    pyset(x=())

Convert `x` to a Python `set`.

If `x` is a Python object, this is equivalent to `set(x)` in Python.
Otherwise `x` must be iterable.
"""
pyset() = pynew(errcheck(C.PySet_New(C.PyNULL)))
pyset(x) = ispy(x) ? pybuiltins.set(x) : pyset_fromiter(x)
export pyset

"""
    pyfrozenset(x=())

Convert `x` to a Python `frozenset`.

If `x` is a Python object, this is equivalent to `frozenset(x)` in Python.
Otherwise `x` must be iterable.
"""
pyfrozenset() = pynew(errcheck(C.PyFrozenSet_New(C.PyNULL)))
pyfrozenset(x) = ispy(x) ? pybuiltins.frozenset(x) : pyfrozenset_fromiter(x)
export pyfrozenset

### dict

pydict_setitem(x::Py, k, v) =
    errcheck(@autopy k v C.PyDict_SetItem(x, k_, v_))

function pydict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        pydict_setitem(ans, k, v)
    end
    return ans
end

function pystrdict_fromiter(kvs)
    ans = pydict()
    for (k, v) in kvs
        pydict_setitem(ans, string(k), v)
    end
    return ans
end

"""
    pydict(x)
    pydict(; x...)

Convert `x` to a Python `dict`. In the second form, the keys are strings.

If `x` is a Python object, this is equivalent to `dict(x)` in Python.
Otherwise `x` must iterate over key-value pairs.
"""
pydict(; kwargs...) =
    isempty(kwargs) ? pynew(errcheck(C.PyDict_New())) : pystrdict_fromiter(kwargs)
pydict(x) = ispy(x) ? pybuiltins.dict(x) : pydict_fromiter(x)
pydict(x::NamedTuple) = pydict(; x...)
export pydict

### datetime

# We used to use 1/1/1 but pandas.Timestamp is a subclass of datetime and does not include
# this date, so we use 1970 instead.
const _base_datetime = DateTime(1970, 1, 1)
const _base_pydatetime = pynew()

function init_datetime()
    pycopy!(_base_pydatetime, pydatetimetype(1970, 1, 1))
end

pydate(year, month, day) = pydatetype(year, month, day)
pydate(x::Date) = pydate(year(x), month(x), day(x))
export pydate

pytime(
    _hour = 0,
    _minute = 0,
    _second = 0,
    _microsecond = 0,
    _tzinfo = nothing;
    hour = _hour,
    minute = _minute,
    second = _second,
    microsecond = _microsecond,
    tzinfo = _tzinfo,
    fold = 0,
) = pytimetype(hour, minute, second, microsecond, tzinfo, fold = fold)
pytime(x::Time) =
    if iszero(nanosecond(x))
        pytime(hour(x), minute(x), second(x), millisecond(x) * 1000 + microsecond(x))
    else
        errset(
            pybuiltins.ValueError,
            "cannot create 'datetime.time' with less than microsecond resolution",
        )
        pythrow()
    end
export pytime

pydatetime(
    year,
    month,
    day,
    _hour = 0,
    _minute = 0,
    _second = 0,
    _microsecond = 0,
    _tzinfo = nothing;
    hour = _hour,
    minute = _minute,
    second = _second,
    microsecond = _microsecond,
    tzinfo = _tzinfo,
    fold = 0,
) = pydatetimetype(year, month, day, hour, minute, second, microsecond, tzinfo, fold = fold)
function pydatetime(x::DateTime)
    # compute time since _base_datetime
    # this accounts for fold
    d = pytimedeltatype(milliseconds = (x - _base_datetime).value)
    ans = _base_pydatetime + d
    pydel!(d)
    return ans
end
pydatetime(x::Date) = pydatetime(year(x), month(x), day(x))
export pydatetime

function pytime_isaware(x)
    tzinfo = pygetattr(x, "tzinfo")
    if pyisnone(tzinfo)
        pydel!(tzinfo)
        return false
    end
    utcoffset = tzinfo.utcoffset
    pydel!(tzinfo)
    o = utcoffset(nothing)
    pydel!(utcoffset)
    ans = !pyisnone(o)
    pydel!(o)
    return ans
end

function pydatetime_isaware(x)
    tzinfo = pygetattr(x, "tzinfo")
    if pyisnone(tzinfo)
        pydel!(tzinfo)
        return false
    end
    utcoffset = tzinfo.utcoffset
    pydel!(tzinfo)
    o = utcoffset(x)
    pydel!(utcoffset)
    ans = !pyisnone(o)
    pydel!(o)
    return ans
end

### fraction

pyfraction(x::Rational) = pyfraction(numerator(x), denominator(x))
pyfraction(x, y) = pyfractiontype(x, y)
pyfraction(x) = pyfractiontype(x)
pyfraction() = pyfractiontype()
export pyfraction

### eval/exec

const MODULE_GLOBALS = Lockable(Dict{Module,Py}())

function _pyeval_args(code, globals, locals)
    if code isa AbstractString
        code_ = code
    elseif ispy(code)
        code_ = code
    else
        throw(ArgumentError("code must be a string or Python code"))
    end
    if globals isa Module
        globals_ = Base.@lock MODULE_GLOBALS get!(pydict, MODULE_GLOBALS[], globals)
    elseif ispy(globals)
        globals_ = globals
    else
        throw(ArgumentError("globals must be a module or a Python dict"))
    end
    if locals === nothing
        locals_ = pynew(Py(globals_))
    elseif ispy(locals)
        locals_ = pynew(Py(locals))
    else
        locals_ = pydict(locals)
    end
    return (code_, globals_, locals_)
end

"""
    pyeval([T=Py], code, globals, locals=nothing)

Evaluate the given Python `code`, returning the result as a `T`.

If `globals` is a `Module`, then a persistent `dict` unique to that module is used.

By default the code runs in global scope (i.e. `locals===globals`). To use a temporary
local scope, set `locals` to `()`, or to a `NamedTuple` of variables to include in the
scope.

See also [`@pyeval`](@ref).

# Examples

The following computes `1.1+2.2` in the `Main` module as a `Float64`:
```
pyeval(Float64, "x+y", Main, (x=1.1, y=2.2))  # returns 3.3
```
"""
function pyeval(::Type{T}, code, globals, locals = nothing) where {T}
    code_, globals_, locals_ = _pyeval_args(code, globals, locals)
    ans = pybuiltins.eval(code_, globals_, locals_)
    pydel!(locals_)
    return pyconvert(T, ans)
end
pyeval(code, globals, locals = nothing) = pyeval(Py, code, globals, locals)
export pyeval

_pyexec_ans(::Type{Nothing}, globals, locals) = nothing
@generated function _pyexec_ans(
    ::Type{NamedTuple{names,types}},
    globals,
    locals,
) where {names,types}
    # TODO: use precomputed interned strings
    # TODO: try to load from globals too
    n = length(names)
    code = []
    vars = Symbol[]
    for i = 1:n
        v = Symbol(:ans, i)
        push!(vars, v)
        push!(
            code,
            :(
                $v = pyconvert(
                    $(types.parameters[i]),
                    pygetitem(locals, $(string(names[i]))),
                )
            ),
        )
    end
    push!(code, :(return $(NamedTuple{names,types})(($(vars...),))))
    return Expr(:block, code...)
end

"""
    pyexec([T=Nothing], code, globals, locals=nothing)

Execute the given Python `code`.

If `globals` is a `Module`, then a persistent `dict` unique to that module is used.

By default the code runs in global scope (i.e. `locals===globals`). To use a temporary
local scope, set `locals` to `()`, or to a `NamedTuple` of variables to include in the
scope.

If `T==Nothing` then returns `nothing`. Otherwise `T` must be a concrete `NamedTuple` type
and the corresponding items from `locals` are extracted and returned.

See also [`@pyexec`](@ref).

# Examples

The following computes `1.1+2.2` in the `Main` module as a `Float64`:
```
pyexec(@NamedTuple{ans::Float64}, "ans=x+y", Main, (x=1.1, y=2.2))  # returns (ans = 3.3,)
```

Marking variables as `global` saves them into the module scope, so that they are available
in subsequent invocations:
```
pyexec("global x; x=12", Main)
pyeval(Int, "x", Main)  # returns 12
```
"""
function pyexec(::Type{T}, code, globals, locals = nothing) where {T}
    code_, globals_, locals_ = _pyeval_args(code, globals, locals)
    pydel!(pybuiltins.exec(code_, globals_, locals_))
    ans = _pyexec_ans(T, globals_, locals_)
    pydel!(locals_)
    return ans
end
pyexec(code, globals, locals = nothing) = pyexec(Nothing, code, globals, locals)
export pyexec

function _pyeval_macro_code(arg)
    if arg isa String
        return arg
    elseif arg isa Expr && arg.head === :macrocall && arg.args[1] == :(`foo`).args[1]
        return arg.args[3]
    else
        return nothing
    end
end

function _pyeval_macro_args(arg, filename, mode)
    # separate out inputs => code => outputs (with only code being required)
    if @capture(arg, inputs_ => code_ => outputs_)
        code = _pyeval_macro_code(code)
        code === nothing && error("invalid code")
    elseif @capture(arg, lhs_ => rhs_)
        code = _pyeval_macro_code(lhs)
        if code === nothing
            code = _pyeval_macro_code(rhs)
            code === nothing && error("invalid code")
            inputs = lhs
            outputs = nothing
        else
            inputs = nothing
            outputs = rhs
        end
    else
        code = _pyeval_macro_code(arg)
        code === nothing && error("invalid code")
        inputs = outputs = nothing
    end
    # precompile the code
    codestr = code
    codeobj = pynew()
    codeready = Ref(false)
    code = quote
        if !$codeready[]
            $pycopy!($codeobj, $pybuiltins.compile($codestr, $filename, $mode))
            $codeready[] = true
        end
        $codeobj
    end
    # convert inputs to locals
    if inputs === nothing
        locals = ()
    else
        if inputs isa Expr && inputs.head === :tuple
            inputs = inputs.args
        else
            inputs = [inputs]
        end
        locals = []
        for input in inputs
            if @capture(input, var_Symbol)
                push!(locals, var => var)
            elseif @capture(input, var_Symbol = ex_)
                push!(locals, var => ex)
            else
                error("invalid input: $input")
            end
        end
        locals = :(($([:($var = $ex) for (var, ex) in locals]...),))
    end
    # done
    return locals, code, outputs
end

"""
    @pyeval [inputs =>] code [=> T]

Evaluate the given `code` in a new local scope and return the answer as a `T`.

The global scope is persistent and unique to the current module.

The `code` must be a literal string or command.

The `inputs` is a tuple of inputs of the form `v=expr` to be included in the local scope.
Only `v` is required, `expr` defaults to `v`.

# Examples

The following computes `1.1+2.2` and returns a `Float64`:
```
@pyeval (x=1.1, y=2.2) => `x+y` => Float64  # returns 3.3
```
"""
macro pyeval(arg)
    locals, code, outputs =
        _pyeval_macro_args(arg, "$(__source__.file):$(__source__.line)", "eval")
    if outputs === nothing
        outputs = Py
    end
    esc(:($pyeval($outputs, $code, $__module__, $locals)))
end
export @pyeval

"""
    @pyexec [inputs =>] code [=> outputs]

Execute the given `code` in a new local scope.

The global scope is persistent and unique to the current module.

The `code` must be a literal string or command.

The `inputs` is a tuple of inputs of the form `v=expr` to be included in the local scope.
Only `v` is required, `expr` defaults to `v`.

The `outputs` is a tuple of outputs of the form `x::T=v`, meaning that `v` is extracted from
locals, converted to `T` and assigned to `x`. Only `x` is required: `T` defaults to `Py`
and `v` defaults to `x`.

# Examples

The following computes `1.1+2.2` and assigns its value to `ans` as a `Float64`:
```
@pyexec (x=1.1, y=2.2) => `ans=x+y` => ans::Float64  # returns 3.3
```

Marking variables as `global` saves them into the module scope, so that they are available
in subsequent invocations:
```
@pyexec `global x; x=12`
@pyeval `x` => Int  # returns 12
```
"""
macro pyexec(arg)
    locals, code, outputs =
        _pyeval_macro_args(arg, "$(__source__.file):$(__source__.line)", "exec")
    if outputs === nothing
        outputs = Nothing
        esc(:($pyexec(Nothing, $code, $__module__, $locals)))
    else
        if outputs isa Expr && outputs.head === :tuple
            oneoutput = false
            outputs = outputs.args
        else
            oneoutput = true
            outputs = [outputs]
        end
        pyvars = Symbol[]
        jlvars = Symbol[]
        types = []
        for output in outputs
            if @capture(output, lhs_ = rhs_)
                rhs isa Symbol || error("invalid output: $output")
                output = lhs
                pyvar = rhs
            else
                pyvar = missing
            end
            if @capture(output, lhs_::rhs_)
                outtype = rhs
                output = lhs
            else
                outtype = Py
            end
            output isa Symbol || error("invalid output: $output")
            if pyvar === missing
                pyvar = output
            end
            push!(pyvars, pyvar)
            push!(jlvars, output)
            push!(types, outtype)
        end
        outtype = :($NamedTuple{($(map(QuoteNode, pyvars)...),),Tuple{$(types...)}})
        ans = :($pyexec($outtype, $code, $__module__, $locals))
        if oneoutput
            ans = :($(jlvars[1]) = $ans[1])
        else
            if pyvars != jlvars
                outtype2 =
                    :($NamedTuple{($(map(QuoteNode, jlvars)...),),Tuple{$(types...)}})
                ans = :($outtype2($ans))
            end
            ans = :(($(jlvars...),) = $ans)
        end
        esc(ans)
    end
end
export @pyexec

### with

"""
    pywith(f, o, d=nothing)

Equivalent to `with o as x: f(x)` in Python, where `x` is a `Py`.

On success, the value of `f(x)` is returned.

If an exception occurs but is suppressed then `d` is returned.
"""
function pywith(f, o, d = nothing)
    o = Py(o)
    t = pytype(o)
    exit = t.__exit__
    value = t.__enter__(o)
    exited = false
    try
        return f(value)
    catch exc
        if exc isa PyException
            exited = true
            if pytruth(exit(o, exc.t, exc.v, exc.b))
                return d
            end
        end
        rethrow()
    finally
        exited || exit(o, pybuiltins.None, pybuiltins.None, pybuiltins.None)
    end
end
export pywith

### import

"""
    pyimport(m)
    pyimport(m => k)
    pyimport(m => (k1, k2, ...))
    pyimport(m1, m2, ...)

Import a module `m`, or an attribute `k`, or a tuple of attributes.

If several arguments are given, return the results of importing each one in a tuple.
"""
pyimport(m) = pynew(errcheck(@autopy m C.PyImport_Import(m_)))
pyimport((m, k)::Pair) = (m_ = pyimport(m); k_ = pygetattr(m_, k); pydel!(m_); k_)
pyimport((m, ks)::Pair{<:Any,<:Tuple}) =
    (m_ = pyimport(m); ks_ = map(k -> pygetattr(m_, k), ks); pydel!(m_); ks_)
pyimport(m1, m2, ms...) = map(pyimport, (m1, m2, ms...))
export pyimport

### builtins not covered elsewhere

"""
    pyprint(...)

Equivalent to `print(...)` in Python.
"""
pyprint(args...; kwargs...) = (pydel!(pybuiltins.print(args...; kwargs...)); nothing)
export pyprint

function _pyhelp(args...)
    pyisnone(pybuiltins.help) && error("Python help is not available")
    pydel!(pybuiltins.help(args...))
    nothing
end
"""
    pyhelp([x])

Equivalent to `help(x)` in Python.
"""
pyhelp() = _pyhelp()
pyhelp(x) = _pyhelp(x)
export pyhelp

"""
    pyall(x)

Equivalent to `all(x)` in Python.
"""
function pyall(x)
    y = pybuiltins.all(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pyall

"""
    pyany(x)

Equivalent to `any(x)` in Python.
"""
function pyany(x)
    y = pybuiltins.any(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pyany

"""
    pycallable(x)

Equivalent to `callable(x)` in Python.
"""
function pycallable(x)
    y = pybuiltins.callable(x)
    z = pybool_asbool(y)
    pydel!(y)
    z
end
export pycallable

"""
    pycompile(...)

Equivalent to `compile(...)` in Python.
"""
pycompile(args...; kwargs...) = pybuiltins.compile(args...; kwargs...)
export pycompile
