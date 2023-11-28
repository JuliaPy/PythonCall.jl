"""
    module PythonCall.Convert

Implements `pyconvert`.
"""
module Convert

using ..C
using ..Core
using ..Core: pyisTrue, pyisFalse, pyfloat_asdouble, pybool_asbool, pystr_asstring, iserrset, iserrset_ambig, errclear, errmatches

import ..Core: pyconvert

export pyconvert

function success(::Type{T}, x::T) where {T}
    Some{T}(x)
end

function fail(::Type)
    nothing
end

function issuccess(x)
    x !== nothing
end

function value(x::Some)
    something(x)
end

"""
    pyconvert(T, x)

Convert the Python object `x` to a `T`.

See [`rule`](@ref) for how to add new conversion rules.
"""
function pyconvert(::Type{T}, x) where {T}
    if T != Union{}
        x = Py(x)
        t = pytype(x)
        for b in _mro(t)
            v = rule(T, Val(b), x)::Union{Some{T},Nothing}
            issuccess(v) && return value(v)::T
        end
    end
    error("could not convert this Python '$(t.__name__)' to a Julia '$T'")
end

function _mro(t::Py)
    [_fullname(b) for b in t.__mro__]
end

function _fullname(b::Py)
    Symbol("$(b.__module__):$(b.__qualname__)")
end

function tryconvert(::Type{T}, x) where {T}
    try
        success(T, convert(T, x))
    catch
        fail(T)
    end
end

"""
    rule(::Type{T}, ::Val{Symbol("module:type")}, x::Py)

Implements a conversion rule trying to convert `x` to a `T`.
"""
function rule(::Type, ::Val, ::Py)
    nothing
end

function rule(::Type{T}, ::Val{Symbol("builtins:object")}, x::Py) where {T}
    if Py <: T
        return success(T, x)
    end
    fail(T)
end

function rule(::Type{T}, ::Val{Symbol("builtins:NoneType")}, x::Py) where {T}
    if Nothing <: T
        return success(T, nothing)
    end
    if Missing <: T
        return success(T, missing)
    end
    fail(T)
end

function rule(::Type{T}, ::Val{Symbol("builtins:bool")}, x::Py) where {T}
    if Bool <: T
        v = pybool_asbool(x)::Bool
        return success(T, v)
    end
    # we don't need to handle conversion to other numeric types because that's handled
    # by the rule for builtins:int, which is a supertype of builtins:bool
    fail(T)
end

function rule(::Type{T}, ::Val{Symbol("builtins:int")}, x::Py) where {T<:Integer}
    TS = typeintersect(T, Signed)
    TU = typeintersect(T, Unsigned)
    if TS != Union{}
        v = C.PyLong_AsLongLong(x)::Clonglong
    elseif TU != Union{}
        v = C.PyLong_AsUnsignedLongLong(x)::Culonglong
    else
        v = C.PyLong_AsLongLong(x)::Clonglong
    end
    if !iserrset_ambig(v)
        # we have the value
        if Int <: T
            ans = tryconvert(Int, v)
            issuccess(ans) && return success(T, value(ans))
        end
        if BigInt <: T
            return success(T, convert(BigInt, v))
        end
        if UInt <: T
            ans = tryconvert(UInt, v)
            issuccess(ans) && return success(T, value(ans))
        end
        return tryconvert(T, v)
    elseif errmatches(pybuiltins.OverflowError)
        # doesn't fit in a native integer type
        # so convert via a string
        errclear()
        hex = pystr_asstring(pybuiltins.hex(x))
        v = parse(BigInt, hex)
        if Int <: T
            ans = tryconvert(Int, v)
            issuccess(ans) && return success(T, value(ans))
        end
        if BigInt <: T
            return success(T, convert(BigInt, v))
        end
        if UInt <: T
            ans = tryconvert(UInt, v)
            issuccess(ans) && return success(T, value(ans))
        end
        return tryconvert(T, v)
    else
        pythrow()
    end
end

function rule(::Type{T}, P::Val{Symbol("builtins:int")}, x::Py) where {T}
    T1 = typeintersect(T, Integer)
    if !(T1 <: Union{})
        ans = rule(T1, P, x)
        issuccess(ans) && return success(T, value(ans))
    end
    T2 = typeintersect(T, Rational)
    T3 = typeintersect(T, Real)
    T4 = typeintersect(T, Number)
    if !(T4 <: Integer)
        v = value(rule(Union{Int,BigInt}, P, x))
        if !(T2 <: Integer)
            ans = tryconvert(T2, v)
            issuccess(ans) && return success(T, value(ans))
        end
        if !(T3 <: Union{Integer,Rational})
            ans = tryconvert(T3, v)
            issuccess(ans) && return success(T, value(ans))
        end
        if !(T4 <: Real)
            ans = tryconvert(T4, v)
            issuccess(ans) && return success(T, value(ans))
        end
    end
    fail(T)
end

function rule(::Type{T}, ::Val{Symbol("builtins:float")}, x::Py) where {T}
    T2 = typeintersect(T, AbstractFloat)
    T3 = typeintersect(T, Real)
    T4 = typeintersect(T, Number)
    # Number
    if T4 != Union{}
        v = pyfloat_asdouble(x)::Cdouble
        # specific floating point types
        if Cdouble <: T
            return success(T, v)
        elseif Float64 <: T
            return success(T, convert(Float64, v))
        elseif BigFloat <: T
            return success(T, convert(BigFloat, v))
        elseif Float32 <: T
            return success(T, convert(Float32, v))
        elseif Float16 <: T
            return success(T, convert(Float16, v))
        end
        # AbstractFloat
        if !(T2 <: Union{Cdouble,Float64,BigFloat,Float32,Float16})
            ans = tryconvert(T2, v)
            issuccess(ans) && return success(T, value(ans))
        end
        # Real
        if !(T3 <: AbstractFloat)
            ans = tryconvert(T3, v)
            issuccess(ans) && return success(T, value(ans))
        end
        # Number
        if !(T4 <: Real)
            ans = tryconvert(T4, v)
            issuccess(ans) && return success(T, value(ans))
        end
    end
    # Missing (from NaN)
    if Missing <: T && isnan(pyfloat_asdouble(x))
        success(T, missing)
    end
    fail(T)
end

function rule(::Type{T}, ::Val{Symbol("builtins:str")}, x::Py) where {T}
    # AbstractString
    T2 = typeintersect(T, AbstractString)
    if T2 != Union{}
        v = pystr_asstring(x)::String
        # String
        if String <: T2
            return success(T, v)
        end
        # AbstractString
        if !(T2 <: String)
            ans = tryconvert(T2, v)
            issuccess(ans) && return success(T, value(ans))
        end
    end
    # AbstractChar
    T3 = typeintersect(T, AbstractChar)
    if T3 != Union{} && pylen(x) == 1
        v = pystr_asstring(x)::String
        # Char
        if Char <: T3
            return success(T, v[1]::Char)
        end
        # AbstractChar
        if !(T3 <: Char)
            ans = tryconvert(T3, v)
            issuccess(ans) && return success(T, value(ans))
        end
    end
    # Symbol
    if Symbol <: T
        v = pystr_asstring(x)::String
        return success(T, Symbol(v)::Symbol)
    end
    fail(T)
end

end
