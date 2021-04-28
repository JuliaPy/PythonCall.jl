module CPython

using Libdl, Dates, Compat, Requires, BangBang
import ..PythonCall:
    CONFIG,
    isnull,
    ism1,
    PYERR,
    NOTIMPLEMENTED,
    _typeintersect,
    _type_lb,
    _type_ub,
    _type_union_split,
    tryconvert,
    ispyreftype,
    ispyref,
    pyptr,
    putresult,
    takeresult,
    CACHE,
    PythonCall
using Base: @kwdef
using UnsafePointers: UnsafePtr

pyglobal(name) = dlsym(CONFIG.libptr, name)
pyloadglobal(::Type{T}, name) where {T} = unsafe_load(Ptr{T}(pyglobal(name)))
pyglobal(r::Ref{Ptr{T}}, name) where {T} = begin
    p = r[]
    if isnull(p)
        p = r[] = Ptr{T}(pyglobal(name))
    end
    p
end
pyloadglobal(r::Ref{Ptr{T}}, name) where {T} = begin
    p = r[]
    if isnull(p)
        p = r[] = unsafe_load(Ptr{Ptr{T}}(pyglobal(name)))
    end
    p
end

include("consts.jl")
include("pointers.jl")
include("fundamentals.jl")
include("none.jl")
include("object.jl")
include("sequence.jl")
include("mapping.jl")
include("method.jl")
include("str.jl")
include("bytes.jl")
include("tuple.jl")
include("type.jl")
include("number.jl")
include("iter.jl")
include("bool.jl")
include("int.jl")
include("float.jl")
include("complex.jl")
include("list.jl")
include("dict.jl")
include("set.jl")
include("buffer.jl")
include("stdlib.jl")
include("collections.jl")
include("io.jl")
include("range.jl")
include("ctypes.jl")
include("numpy.jl")
include("slice.jl")
include("fraction.jl")
include("datetime.jl")
include("newtype.jl")
include("juliaerror.jl")
include("juliabase.jl")
include("juliaraw.jl")
include("juliaany.jl")
include("juliaiterator.jl")
include("juliatype.jl")
include("juliadict.jl")
include("juliaset.jl")
include("juliaarray.jl")
include("juliavector.jl")
include("juliamodule.jl")
include("julianumber.jl")
include("juliaio.jl")
include("as.jl")
include("arg.jl")

init() = begin
    PyObject_TryConvert_AddRule("builtins.NoneType", Nothing, PyNone_TryConvertRule_nothing, 100)
    PyObject_TryConvert_AddRule("builtins.NoneType", Missing, PyNone_TryConvertRule_missing)
    PyObject_TryConvert_AddRule("builtins.bool", Bool, PyBool_TryConvertRule_bool, 100)
    PyObject_TryConvert_AddRule("numbers.Integral", Integer, PyLongable_TryConvertRule_integer, 100)
    PyObject_TryConvert_AddRule("numbers.Integral", Rational, PyLongable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Integral", Real, PyLongable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Integral", Number, PyLongable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Integral", Any, PyLongable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("builtins.float", Float64, PyFloatable_TryConvertRule_convert, 100)
    PyObject_TryConvert_AddRule("numbers.Real", Float64, PyFloatable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Real", BigFloat, PyFloatable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Real", Float32, PyFloatable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Real", Float16, PyFloatable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Real", AbstractFloat, PyFloatable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Real", Real, PyFloatable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Real", Number, PyFloatable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Real", Nothing, PyFloatable_TryConvertRule_nothing)
    PyObject_TryConvert_AddRule("numbers.Real", Missing, PyFloatable_TryConvertRule_missing)
    PyObject_TryConvert_AddRule("builtins.complex", Complex{Float64}, PyComplexable_TryConvertRule_convert, 100)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{Float64}, PyComplexable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{BigFloat}, PyComplexable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{Float32}, PyComplexable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{Float16}, PyComplexable_TryConvertRule_convert)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{T} where {T<:AbstractFloat}, PyComplexable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Complex", Complex{T} where {T<:Real}, PyComplexable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("numbers.Complex", Number, PyComplexable_TryConvertRule_tryconvert)
    PyObject_TryConvert_AddRule("builtins.bytes", Vector{UInt8}, PyBytes_TryConvertRule_vector)
    PyObject_TryConvert_AddRule("builtins.bytes", Vector{Int8}, PyBytes_TryConvertRule_vector)
    PyObject_TryConvert_AddRule("builtins.bytes", String, PyBytes_TryConvertRule_string)
    PyObject_TryConvert_AddRule("builtins.str", String, PyUnicode_TryConvertRule_string, 100)
    PyObject_TryConvert_AddRule("builtins.str", Symbol, PyUnicode_TryConvertRule_symbol)
    PyObject_TryConvert_AddRule("builtins.str", Char, PyUnicode_TryConvertRule_char)
    PyObject_TryConvert_AddRule("builtins.str", Vector{UInt8}, PyUnicode_TryConvertRule_vector)
    PyObject_TryConvert_AddRule("builtins.str", Vector{Int8}, PyUnicode_TryConvertRule_vector)
    PyObject_TryConvert_AddRule("builtins.tuple", Tuple, PyIterable_ConvertRule_tuple, 100)
    PyObject_TryConvert_AddRule("builtins.range", StepRange{T,S} where {T<:Integer,S<:Integer}, PyRange_TryConvertRule_steprange, 100)
    PyObject_TryConvert_AddRule("builtins.range", UnitRange{T} where {T<:Integer}, PyRange_TryConvertRule_unitrange)
    PyObject_TryConvert_AddRule("collections.abc.Iterable", Vector, PyIterable_ConvertRule_vecorset)
    PyObject_TryConvert_AddRule("collections.abc.Iterable", Set, PyIterable_ConvertRule_vecorset)
    PyObject_TryConvert_AddRule("collections.abc.Iterable", Tuple, PyIterable_ConvertRule_tuple)
    PyObject_TryConvert_AddRule("collections.abc.Iterable", NamedTuple, PyIterable_ConvertRule_namedtuple)
    PyObject_TryConvert_AddRule("collections.abc.Iterable", Pair, PyIterable_ConvertRule_pair)
    PyObject_TryConvert_AddRule("collections.abc.Sequence", Vector, PyIterable_ConvertRule_vecorset)
    PyObject_TryConvert_AddRule("collections.abc.Set", Set, PyIterable_ConvertRule_vecorset)
    PyObject_TryConvert_AddRule("collections.abc.Mapping", Dict, PyMapping_ConvertRule_dict)
    PyObject_TryConvert_AddRule("datetime.time", Time, PyTime_TryConvertRule_time, 100)
    PyObject_TryConvert_AddRule("datetime.date", Date, PyDate_TryConvertRule_date, 100)
    PyObject_TryConvert_AddRule("datetime.datetime", DateTime, PyDateTime_TryConvertRule_datetime, 100)
    PyObject_TryConvert_AddRule("datetime.timedelta", Dates.Period, PyTimeDelta_TryConvertRule_period, 100)
    PyObject_TryConvert_AddRule("datetime.timedelta", Dates.CompoundPeriod, PyTimeDelta_TryConvertRule_compoundperiod)
    PyObject_TryConvert_AddRule("juliacall.ValueBase", Any, PyJuliaValue_TryConvert_any, 1000)
    PyObject_TryConvert_AddRule("juliacall.As", Any, PyAs_ConvertRule_tryconvert, 1000)
    PyObject_TryConvert_AddExtraType(PyIterableABC_Type)
    PyObject_TryConvert_AddExtraType(PyCallableABC_Type)
    PyObject_TryConvert_AddExtraType(PySequenceABC_Type)
    PyObject_TryConvert_AddExtraType(PyMappingABC_Type)
    PyObject_TryConvert_AddExtraType(PySetABC_Type)
    PyObject_TryConvert_AddExtraType(PyNumberABC_Type)
    PyObject_TryConvert_AddExtraType(PyComplexABC_Type)
    PyObject_TryConvert_AddExtraType(PyRealABC_Type)
    PyObject_TryConvert_AddExtraType(PyRationalABC_Type)
    PyObject_TryConvert_AddExtraType(PyIntegralABC_Type)

    ### ctypes
    for (p, T) in [
        ("char", Cchar),
        ("wchar", Cwchar_t),
        ("byte", Cchar),
        ("ubyte", Cuchar),
        ("short", Cshort),
        ("ushort", Cushort),
        ("int", Cint),
        ("uint", Cuint),
        ("long", Clong),
        ("ulong", Culong),
        ("longlong", Clonglong),
        ("ulonglong", Culonglong),
        ("size_t", Csize_t),
        ("ssize_t", Cssize_t),
        ("float", Cfloat),
        ("double", Cdouble), #=("longdouble", ???),=#
        ("char_p", Ptr{Cchar}),
        ("wchar_p", Ptr{Cwchar_t}),
        ("void_p", Ptr{Cvoid}),
    ]
        isptr = occursin("_p", p)
        isfloat = occursin("float", p) || occursin("double", p)
        isint = !(isfloat || isptr)
        isreal = isint || isfloat
        p == "char_p" && PyObject_TryConvert_AddRule("ctypes.c_$p", Cstring, PySimpleCData_TryConvert_value{T,false}())
        p == "wchar_p" && PyObject_TryConvert_AddRule("ctypes.c_$p", Cwstring, PySimpleCData_TryConvert_value{T,false}())
        PyObject_TryConvert_AddRule("ctypes.c_$p", T, PySimpleCData_TryConvert_value{T,false}())
        isint && PyObject_TryConvert_AddRule("ctypes.c_$p", Integer, PySimpleCData_TryConvert_value{T,true}())
        isint && PyObject_TryConvert_AddRule("ctypes.c_$p", Rational, PySimpleCData_TryConvert_value{T,true}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", Float64, PySimpleCData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", BigFloat, PySimpleCData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", Float32, PySimpleCData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", Float16, PySimpleCData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", AbstractFloat, PySimpleCData_TryConvert_value{T,true}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", Real, PySimpleCData_TryConvert_value{T,true}())
        isreal && PyObject_TryConvert_AddRule("ctypes.c_$p", Number, PySimpleCData_TryConvert_value{T,true}())
        isptr && PyObject_TryConvert_AddRule("ctypes.c_$p", Ptr, PySimpleCData_TryConvert_value{T,false}())
        PyObject_TryConvert_AddRule("ctypes.c_$p", Any, PySimpleCData_TryConvert_value{T,true}())
    end

    ### numpy
    # TODO: Compound types
    # TODO: datetime64, timedelta64
    for (p, T) in [
        ("int8", Int8),
        ("int16", Int16),
        ("int32", Int32),
        ("int64", Int64),
        ("int128", Int128),
        ("uint8", UInt8),
        ("uint16", UInt16),
        ("uint32", UInt32),
        ("uint64", UInt64),
        ("uint128", UInt128),
        ("float16", Float16),
        ("float32", Float32),
        ("float64", Float64),
        ("complex32", Complex{Float16}),
        ("complex64", Complex{Float32}),
        ("complex128", Complex{Float64}),
    ]
        isint = occursin("int", p)
        isfloat = occursin("float", p)
        iscomplex = occursin("complex", p)
        isreal = isint || isfloat
        PyObject_TryConvert_AddRule("numpy.$p", T, PyNumpySimpleData_TryConvert_value{T,false}(), 100)
        isint && PyObject_TryConvert_AddRule("numpy.$p", Integer, PyNumpySimpleData_TryConvert_value{T,true}())
        isint && PyObject_TryConvert_AddRule("numpy.$p", Rational, PyNumpySimpleData_TryConvert_value{T,true}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", Float64, PyNumpySimpleData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", BigFloat, PyNumpySimpleData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", Float32, PyNumpySimpleData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", Float16, PyNumpySimpleData_TryConvert_value{T,false}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", AbstractFloat, PyNumpySimpleData_TryConvert_value{T,true}())
        isreal && PyObject_TryConvert_AddRule("numpy.$p", Real, PyNumpySimpleData_TryConvert_value{T,true}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{Float64}, PyNumpySimpleData_TryConvert_value{T,false}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{BigFloat}, PyNumpySimpleData_TryConvert_value{T,false}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{Float32}, PyNumpySimpleData_TryConvert_value{T,false}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{Float16}, PyNumpySimpleData_TryConvert_value{T,false}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{T} where {T<:AbstractFloat}, PyNumpySimpleData_TryConvert_value{T,true}())
        iscomplex && PyObject_TryConvert_AddRule("numpy.$p", Complex{T} where {T<:Real}, PyNumpySimpleData_TryConvert_value{T,true}())
        PyObject_TryConvert_AddRule("numpy.$p", Number, PyNumpySimpleData_TryConvert_value{T,true}())
        PyObject_TryConvert_AddRule("numpy.$p", Any, PyNumpySimpleData_TryConvert_value{T,true}())
    end
end
precompile(init, ())
@init init()

precompile(PyJuliaValue_New, (PyPtr, Function))
precompile(PyMappingABC_Type, ())

end
