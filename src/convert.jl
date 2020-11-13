function pyconvert(::Type{T}, o::AbstractPyObject) where {T}
    r = pytryconvert(T, o)
    r === PyConvertFail() ? error("cannot convert this Python `$(pytype(o).__name__)` to a Julia `$T`") : r
end
export pyconvert

function pytryconvert(::Type{T}, o::AbstractPyObject) where {T}
    # special cases
    if T == PyObject
        return PyObject(o)
    end

    # types
    for t in pytype(o).__mro__
        n = "$(t.__module__).$(t.__name__)"
        c = get(PYTRYCONVERT_TYPE_RULES, n, (T,o)->PyConvertFail())
        r = c(T, o) :: Union{T, PyConvertFail}
        r === PyConvertFail() || return r
    end

    # interfaces
    # TODO: buffer
    # TODO: IO
    # TODO: number?
    if pyisinstance(o, pyiterableabc)
        r = pyiterable_tryconvert(T, o) :: Union{T, PyConvertFail}
        r === PyConvertFail() || return r
    end

    # so that T=Any always succeeds
    if o isa T
        return o
    end

    # failure
    return PyConvertFail()
end
export pytryconvert

const PYTRYCONVERT_TYPE_RULES = Dict{String,Function}(
    "builtins.NoneType" => pynone_tryconvert,
    "builtins.bool" => pybool_tryconvert,
    "builtins.str" => pystr_tryconvert,
    "builtins.int" => pyint_tryconvert,
    "builtins.float" => pyfloat_tryconvert,
    "builtins.complex" => pycomplex_tryconvert,
    "builtins.Fraction" => pyfraction_tryconvert,
    "builtins.range" => pyrange_tryconvert,
    "builtins.tuple" => pytuple_tryconvert,
    "pandas.core.frame.DataFrame" => pypandasdataframe_tryconvert,
    # TODO: datetime, date, time
    # NOTE: we don't need to include standard containers here because we can access them via standard interfaces (Sequence, Mapping, Buffer, etc.)
)

Base.convert(::Type{T}, o::AbstractPyObject) where {T} = pyconvert(T, o)
Base.convert(::Type{Any}, o::AbstractPyObject) = o

### SPECIAL CONVERSIONS

@generated _eltype(o) = try eltype(o); catch; missing; end

@generated _keytype(o) = try keytype(o); catch; missing; end
_keytype(o::Base.RefValue) = Tuple{}
_keytype(o::NamedTuple) = Union{Symbol,Int}
_keytype(o::Tuple) = Int

@generated _valtype(o, k...) = try valtype(o); catch; missing; end
_valtype(o::NamedTuple, k::Int) = fieldtype(typeof(o), k)
_valtype(o::NamedTuple, k::Symbol) = fieldtype(typeof(o), k)
_valtype(o::Tuple, k::Int) = fieldtype(typeof(o), k)

hasmultiindex(o) = true
hasmultiindex(o::AbstractDict) = false
hasmultiindex(o::NamedTuple) = false
hasmultiindex(o::Tuple) = false

"""
    pyconvert_element(o, v::AbstractPyObject)

Convert `v` to be of the right type to be an element of `o`.
"""
pytryconvert_element(o, v) = pytryconvert(_eltype(o)===missing ? Any : _eltype(o), v)
pyconvert_element(args...) =
    let r = pytryconvert_element(args...)
        r === PyConvertFail() ? error("cannot convert this to an element") : r
    end

"""
    pytryconvert_indices(o, k::AbstractPyObject)

Convert `k` to be of the right type to be a tuple of indices for `o`.
"""
function pytryconvert_indices(o, k)
    if _keytype(o) !== missing
        i = pytryconvert(_keytype(o), k)
        i === PyConvertFail() ? PyConvertFail() : (i,)
    elseif hasmultiindex(o) && pyistuple(k)
        Tuple(pyconvert(Any, x) for x in k)
    else
        (pyconvert(Any, k),)
    end
end
pyconvert_indices(args...) =
    let r = pytryconvert_indices(args...)
        r === PyConvertFail() ? error("cannot convert this to indices") : r
    end

"""
    pyconvert_value(o, v::AbstractPyObject, k...)

Convert `v` to be of the right type to be a value of `o` at indices `k`.
"""
pytryconvert_value(o, v, k...) = pytryconvert(_valtype(o)===missing ? Any : _valtype(o), v)
pyconvert_value(args...) =
    let r = pytryconvert_value(args...)
        r === PyConvertFail() ? error("cannot convert this to a value") : r
    end
