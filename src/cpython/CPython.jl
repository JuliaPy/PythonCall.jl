module CPython

using Libdl
import ..Python: CONFIG, isnull, ism1, PYERR, NOTIMPLEMENTED, _typeintersect, tryconvert, ispyreftype, pyptr, putresult, takeresult, moveresult, CACHE, Python
using Base: @kwdef
using UnsafePointers: UnsafePtr

pyglobal(name) = dlsym(CONFIG.libptr, name)
pyglobal(r::Ref{Ptr{T}}, name) where {T} = (p=r[]; if isnull(p); p=r[]=Ptr{T}(pyglobal(name)); end; p)
pyloadglobal(r::Ref{Ptr{T}}, name) where {T} = (p=r[]; if isnull(p); p=r[]=unsafe_load(Ptr{Ptr{T}}(pyglobal(name))) end; p)

macro cdef(name, rettype, argtypes)
    name isa QuoteNode && name.value isa Symbol || error("name must be a symbol, got $name")
    jname = esc(name.value)
    refname = esc(Symbol(name.value, :__ref))
    name = esc(name)
    rettype = esc(rettype)
    argtypes isa Expr && argtypes.head==:tuple || error("argtypes must be a tuple, got $argtypes")
    nargs = length(argtypes.args)
    argtypes = esc(argtypes)
    args = [gensym() for i in 1:nargs]
    quote
        const $refname = Ref(C_NULL)
        $jname($(args...)) = ccall(pyglobal($refname, $name), $rettype, $argtypes, $(args...))
    end
end

include("consts.jl")
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
include("collections.jl")
include("range.jl")
include("ctypes.jl")
include("numpy.jl")

__init__() = begin
    PyObject_TryConvert_AddRules("builtins.NoneType", [
        (Nothing, PyNone_TryConvertRule_nothing, 100),
        (Missing, PyNone_TryConvertRule_missing),
    ])
    PyObject_TryConvert_AddRules("builtins.bool", [
        (Bool, PyBool_TryConvertRule_bool, 100),
    ])
    PyObject_TryConvert_AddRules("numbers.Integral", [
        (Integer, PyLongable_TryConvertRule_integer, 100),
        (Rational, PyLongable_TryConvertRule_tryconvert),
        (Real, PyLongable_TryConvertRule_tryconvert),
        (Number, PyLongable_TryConvertRule_tryconvert),
        (Any, PyLongable_TryConvertRule_tryconvert),
    ])
    PyObject_TryConvert_AddRules("builtins.float", [
        (Float64, PyFloatable_TryConvertRule_convert, 100)
    ])
    PyObject_TryConvert_AddRules("numbers.Real", [
        (Float64, PyFloatable_TryConvertRule_convert),
        (BigFloat, PyFloatable_TryConvertRule_convert),
        (Float32, PyFloatable_TryConvertRule_convert),
        (Float16, PyFloatable_TryConvertRule_convert),
        (AbstractFloat, PyFloatable_TryConvertRule_tryconvert),
        (Real, PyFloatable_TryConvertRule_tryconvert),
        (Number, PyFloatable_TryConvertRule_tryconvert),
    ])
    PyObject_TryConvert_AddRules("builtins.complex", [
        (Complex{Float64}, PyComplexable_TryConvertRule_convert, 100),
    ])
    PyObject_TryConvert_AddRules("numbers.Complex", [
        (Complex{Float64}, PyComplexable_TryConvertRule_convert),
        (Complex{BigFloat}, PyComplexable_TryConvertRule_convert),
        (Complex{Float32}, PyComplexable_TryConvertRule_convert),
        (Complex{Float16}, PyComplexable_TryConvertRule_convert),
        (Complex{T} where {T<:AbstractFloat}, PyComplexable_TryConvertRule_tryconvert),
        (Complex{T} where {T<:Real}, PyComplexable_TryConvertRule_tryconvert),
        (Number, PyComplexable_TryConvertRule_tryconvert),
    ])
    PyObject_TryConvert_AddRules("builtins.bytes", [
        (Vector{UInt8}, PyBytes_TryConvertRule_vector),
        (Vector{Int8}, PyBytes_TryConvertRule_vector),
        (String, PyBytes_TryConvertRule_string),
    ])
    PyObject_TryConvert_AddRules("builtins.str", [
        (String, PyUnicode_TryConvertRule_string, 100),
        (Symbol, PyUnicode_TryConvertRule_symbol),
        (Char, PyUnicode_TryConvertRule_char),
        (Vector{UInt8}, PyUnicode_TryConvertRule_vector),
        (Vector{Int8}, PyUnicode_TryConvertRule_vector),
    ])
    PyObject_TryConvert_AddRules("builtins.tuple", [
        (Tuple, PyIterable_ConvertRule_tuple, 100),
    ])
    PyObject_TryConvert_AddRules("builtins.range", [
        (StepRange{T,S} where {T<:Integer, S<:Integer}, PyRange_TryConvertRule_steprange, 100),
        (UnitRange{T} where {T<:Integer}, PyRange_TryConvertRule_unitrange),
    ])
    PyObject_TryConvert_AddRules("collections.abc.Iterable", [
        (Vector, PyIterable_ConvertRule_vector),
        (Set, PyIterable_ConvertRule_set),
        (Tuple, PyIterable_ConvertRule_tuple),
        (Pair, PyIterable_ConvertRule_pair),
    ])
    PyObject_TryConvert_AddRules("collections.abc.Sequence", [
        (Vector, PyIterable_ConvertRule_vector),
    ])
    PyObject_TryConvert_AddRules("collections.abc.Set", [
        (Set, PyIterable_ConvertRule_set),
    ])
    PyObject_TryConvert_AddRules("collections.abc.Mapping", [
        (Dict, PyMapping_ConvertRule_dict),
    ])
    PyObject_TryConvert_AddExtraTypes([
        PyIterableABC_Type,
        PyCallableABC_Type,
        PySequenceABC_Type,
        PyMappingABC_Type,
        PySetABC_Type,
        PyNumberABC_Type,
        PyComplexABC_Type,
        PyRealABC_Type,
        PyRationalABC_Type,
        PyIntegralABC_Type,
    ])

    ### ctypes
    for (p,T) in [("char", Cchar), ("wchar", Cwchar_t), ("byte", Cchar), ("ubyte", Cuchar),
        ("short", Cshort), ("ushort", Cushort), ("int", Cint), ("uint", Cuint),
        ("long", Clong), ("ulong", Culong), ("longlong", Culonglong), ("size_t", Csize_t),
        ("ssize_t", Cssize_t), ("float", Cfloat), ("double", Cdouble), #=("longdouble", ???),=#
        ("char_p", Ptr{Cchar}), ("wchar_p", Ptr{Cwchar_t}), ("void_p", Ptr{Cvoid})]
        isptr = occursin("_p", p)
        isfloat = occursin("float", p) || occursin("double", p)
        isint = !(isfloat || isptr)
        isreal = isint || isfloat
        PyObject_TryConvert_AddRules("ctypes.c_$p", [
            (p=="char_p" ? Cstring : p=="wchar_p" ? Cwstring : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (T, PySimpleCData_TryConvert_value{T,false}()),
            (isint ? Integer : Union{}, PySimpleCData_TryConvert_value{T,true}()),
            (isint ? Rational : Union{}, PySimpleCData_TryConvert_value{T,true}()),
            (isreal ? Float64 : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (isreal ? BigFloat : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (isreal ? Float32 : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (isreal ? Float16 : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (isreal ? AbstractFloat : Union{}, PySimpleCData_TryConvert_value{T,true}()),
            (isreal ? Real : Union{}, PySimpleCData_TryConvert_value{T,true}()),
            (isreal ? Number : Union{}, PySimpleCData_TryConvert_value{T,true}()),
            (isptr ? Ptr : Union{}, PySimpleCData_TryConvert_value{T,false}()),
            (Any, PySimpleCData_TryConvert_value{T,true}()),
        ])
    end

    ### numpy
    # TODO: Compound types
    # TODO: datetime64, timedelta64
    for (p,T) in [("int8", Int8), ("int16", Int16), ("int32", Int32), ("int64", Int64),
        ("int128", Int128), ("uint8", UInt8), ("uint16", UInt16), ("uint32", UInt32),
        ("uint64", UInt64), ("uint128", UInt128), ("float16", Float16), ("float32", Float32),
        ("float64", Float64), ("complex32", Complex{Float16}),
        ("complex64", Complex{Float32}), ("complex128", Complex{Float64})]
        isint = occursin("int", p)
        isfloat = occursin("float", p)
        iscomplex = occursin("complex", p)
        isreal = isint || isfloat
        PyObject_TryConvert_AddRules("numpy.$p", [
            (T, PyNumpySimpleData_TryConvert_value{T,false}(), 100),
            (isint ? Integer : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (isint ? Rational : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (isreal ? Float64 : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (isreal ? BigFloat : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (isreal ? Float32 : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (isreal ? Float16 : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (isreal ? AbstractFloat : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (isreal ? Real : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (iscomplex ? Complex{Float64} : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (iscomplex ? Complex{BigFloat} : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (iscomplex ? Complex{Float32} : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (iscomplex ? Complex{Float16} : Union{}, PyNumpySimpleData_TryConvert_value{T,false}()),
            (iscomplex ? (Complex{T} where {T<:AbstractFloat}) : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (iscomplex ? (Complex{T} where {T<:Real}) : Union{}, PyNumpySimpleData_TryConvert_value{T,true}()),
            (Number, PyNumpySimpleData_TryConvert_value{T,true}()),
            (Any, PyNumpySimpleData_TryConvert_value{T,true}()),
        ])
    end
end

end
