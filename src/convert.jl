struct PyConvertFail end

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

    # traverse MRO
    for t in pytype(o).__mro__
        n = Symbol(cpytypename(pyptr(t)))
        r = pytryconvert_rule(T, Val(n), o)::Union{T,PyConvertFail}
        r===PyConvertFail() || return r
    end

    # interfaces

    # so that T=Any always succeeds
    if o isa T
        return o
    end

    # failure
    return PyConvertFail()
end
export pytryconvert

pytryconvert_rule(::Type, ::Val, o) = PyConvertFail()
pytryconvert_rule(::Type{T}, ::Val{:NoneType}, o) where {T} = pynone_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:bool}, o) where {T} = pybool_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:str}, o) where {T} = pystr_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:int}, o) where {T} = pyint_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:float}, o) where {T} = pyfloat_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:complex}, o) where {T} = pycomplex_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:Fraction}, o) where {T} = pyfraction_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:range}, o) where {T} = pyrange_tryconvert(T, o)
# TODO: all the following are not implemented
pytryconvert_rule(::Type{T}, ::Val{:datetime}, o) where {T} = pydatetime_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:date}, o) where {T} = pydate_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:time}, o) where {T} = pytime_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:bytes}, o) where {T} = pybytes_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:bytearray}, o) where {T} = pybytearray_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:tuple}, o) where {T} = pytuple_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:list}, o) where {T} = pylist_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:dict}, o) where {T} = pydict_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:set}, o) where {T} = pyset_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:frozenset}, o) where {T} = pyfrozenset_tryconvert(T, o)
pytryconvert_rule(::Type{T}, ::Val{:DataFrame}, o) where {T} = pypandasdataframe_tryconvert(T, o)

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



### TYPE UTILITIES

tryconvert(::Type{T}, x) where {T} =
    try
        convert(T, x)
    catch
        PyConvertFail()
    end
tryconvert(::Type{T}, x::T) where {T} = x
tryconvert(::Type{T}, x::PyConvertFail) where {T} = x

@generated _typeintersect(::Type{T1}, ::Type{T2}) where {T1,T2} = typeintersect(T1, T2)
