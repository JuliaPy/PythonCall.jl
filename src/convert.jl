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

pytryconvert_element(o, v) = pytryconvert(eltype(o), v)
pyconvert_element(args...) =
    let r = pytryconvert_element(args...)
        r === PyConvertFail() ? error("cannot convert this to an element") : r
    end

pytryconvert_key(o, k) = pytryconvert(keytype(o), k)
pyconvert_key(args...) =
    let r = pytryconvert_key(args...)
        r === PyConvertFail() ? error("cannot convert this to a key") : r
    end

pytryconvert_value(o, k, v) = pytryconvert_value(o, v)
pytryconvert_value(o, v) = pytryconvert(valtype(o), v)
pyconvert_value(args...) =
    let r = pytryconvert_value(args...)
        r === PyConvertFail() ? error("cannot convert this to a value") : r
    end
