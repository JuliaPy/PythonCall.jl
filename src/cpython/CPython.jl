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

__init__() = begin
    PyObject_TryConvert_AddRules("builtins.NoneType", [
        (Nothing, PyNone_TryConvertRule_nothing, 100),
        (Missing, PyNone_TryConvertRule_missing),
    ])
    PyObject_TryConvert_AddRules("builtins.bool", [
        (Bool, PyBool_TryConvertRule_bool, 100),
    ])
    PyObject_TryConvert_AddRules("builtins.int", [
        (Integer, PyLong_TryConvertRule_integer, 100),
        (Rational, PyLong_TryConvertRule_tryconvert),
        (Real, PyLong_TryConvertRule_tryconvert),
        (Number, PyLong_TryConvertRule_tryconvert),
        (Any, PyLong_TryConvertRule_tryconvert),
    ])
    PyObject_TryConvert_AddRules("builtins.float", [
        (Float64, PyFloat_TryConvertRule_convert, 100),
        (BigFloat, PyFloat_TryConvertRule_convert),
        (Float32, PyFloat_TryConvertRule_convert),
        (Float16, PyFloat_TryConvertRule_convert),
        (AbstractFloat, PyFloat_TryConvertRule_tryconvert),
        (Real, PyFloat_TryConvertRule_tryconvert),
        (Number, PyFloat_TryConvertRule_tryconvert),
    ])
    PyObject_TryConvert_AddRules("builtins.complex", [
        (Complex{Float64}, PyComplex_TryConvertRule_convert, 100),
        (Complex{BigFloat}, PyComplex_TryConvertRule_convert),
        (Complex{Float32}, PyComplex_TryConvertRule_convert),
        (Complex{Float16}, PyComplex_TryConvertRule_convert),
        (Complex{T} where {T<:AbstractFloat}, PyComplex_TryConvertRule_tryconvert),
        (Complex{T} where {T<:Real}, PyComplex_TryConvertRule_tryconvert),
        (Number, PyComplex_TryConvertRule_tryconvert),
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
        ("collections.abc.Iterable", PyIterableABC_SubclassCheck),
        ("collections.abc.Callable", PyCallableABC_SubclassCheck),
        ("collections.abc.Sequence", PySequenceABC_SubclassCheck),
        ("collections.abc.Mapping", PyMappingABC_SubclassCheck),
        ("collections.abc.Set", PySetABC_SubclassCheck),
        ("builtins.bufferable", PyType_CheckBuffer),
    ])
end

end
