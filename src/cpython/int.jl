@cdef :PyLong_FromLongLong PyPtr (Clonglong,)
@cdef :PyLong_FromUnsignedLongLong PyPtr (Culonglong,)
@cdef :PyLong_FromString PyPtr (Cstring, Ptr{Cvoid}, Cint)
@cdef :PyLong_AsLongLong Clonglong (PyPtr,)
@cdef :PyLong_AsUnsignedLongLong Culonglong (PyPtr,)

const PyLong_Type__ref = Ref(PyPtr())
PyLong_Type() = pyglobal(PyLong_Type__ref, :PyLong_Type)

PyLong_Check(o) = Py_TypeCheckFast(o, Py_TPFLAGS_LONG_SUBCLASS)

PyLong_CheckExact(o) = Py_TypeCheckExact(o, PyLong_Type())

PyLong_From(x::Union{Bool,Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64,Int128,UInt128,BigInt}) =
    if x isa Signed && typemin(Clonglong) ≤ x ≤ typemax(Clonglong)
        PyLong_FromLongLong(x)
    elseif typemin(Culonglong) ≤ x ≤ typemax(Culonglong)
        PyLong_FromUnsignedLongLong(x)
    else
        PyLong_FromString(string(x), C_NULL, 10)
    end

PyLong_From(x::Integer) = begin
    y = tryconvert(BigInt, x)
    y === PYERR() && return PyPtr()
    y === NOTIMPLEMENTED() && (PyErr_SetString(PyExc_NotImplementedError(), "Cannot convert this Julia '$(typeof(x))' to a Python 'int'"); return PyPtr())
    PyLong_From(y::BigInt)
end

PyLong_TryConvertRule_integer(o, ::Type{T}, ::Type{S}) where {T, S<:Integer} = begin
    # first try to convert to Clonglong (or Culonglong if unsigned)
    x = S <: Unsigned ? PyLong_AsUnsignedLongLong(o) : PyLong_AsLongLong(o)
    if !ism1(x) || !PyErr_IsSet()
        # success
        return putresult(T, tryconvert(S, x))
    elseif PyErr_IsSet(PyExc_OverflowError())
        # overflows Clonglong or Culonglong
        PyErr_Clear()
        if S in (Bool,Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128) && typemin(typeof(x)) ≤ typemin(S) && typemax(S) ≤ typemax(typeof(x))
            # definitely overflows S, give up now
            return 0
        else
            # try converting to String then BigInt then S
            so = PyObject_Str(o)
            isnull(so) && return -1
            r = PyUnicode_TryConvertRule_string(so, String, String)
            Py_DecRef(so)
            r == 1 || return r
            y = tryparse(BigInt, takeresult(String))
            y === nothing && (PyErr_SetString(PyExc_ValueError(), "Cannot convert this '$(PyType_Name(Py_Type(o)))' to a Julia 'BigInt' because its string representation cannot be parsed as an integer"); return -1)
            return putresult(T, tryconvert(S, y))
        end
    else
        # other error
        return -1
    end
end

PyLong_TryConvertRule_tryconvert(o, ::Type{T}, ::Type{S}) where {T,S} = begin
    r = PyLong_TryConvertRule_integer(o, Integer, Integer)
    r == 1 ? putresult(T, tryconvert(S, takeresult(Integer))) : r
end
