# RULES TO IMPLEMENT:
# Jl -> Any
# buffers/arrays -> PyArray / AbstractArray
# Integral -> Integer
# float / Real -> Float64 / AbstractFloat
# complex / Complex -> ComplexF64 / Complex
# range -> StepRange / UnitRange
# tuple -> Tuple
# Mapping -> PyDict / AbstractDict
# Sequece -> PyList / AbstractVector
# Set -> PySet / AbstractSet
# Collection -> PyCollection
# Iterable -> PyIterable ??
# IOBase -> PyIO
# BaseException -> PyException
# date -> PyDate / Date
# time -> PyTime / Time
# datetime -> PyDateTime / DateTime
# timedelta -> PyTimeDelta / Period / CompoundPeriod
# numpy intXX / uintXX / floatXX -> corresponding type
# ctypes ints / floats / pointers -> corresponding type
# bytes / bytearray -> Vector{UInt8} / Vector{Int8}

function pyconvert(::Type{T}, x::Py) where {T}
    ans = pytryconvert(T, x)::Union{Some{T},Nothing}
    if ans === nothing
        error("Cannot convert this Python '$(pytype(x).__name__)' to a Julia '$(T)'")
    end
    something(ans)
end

function pyconvert(::Type{T}, x::Any) where {T}
    return pyconvert(T, Py(x)::Py)
end
export pyconvert

function pytryconvert(::Type{T}, x::Py) where {T}
    t = typeinfo(pytype(x))

    # subtypes
    TNumber = Utils._typeintersect(T, Number)
    TAbstractString = Utils._typeintersect(T, AbstractString)
    TAbstractChar = Utils._typeintersect(T, AbstractChar)


    # None -> Nothing / Missing
    if (Nothing <: T || Missing <: T) && t.is_none
        if Nothing <: T
            return Some{T}(nothing)
        end
        if Missing <: T
            return Some{T}(missing)
        end
    end

    # bool -> Bool
    if Bool <: T && t.is_bool
        return Some{T}(pybool_asbool(x))
    end

    # Number -> Number
    if TNumber != Union{} && t.is_abstract_number
        if t.is_abstract_integral
            error("not implemented")
        elseif t.is_abstract_rational
            error("not implemented")
        elseif t.is_abstract_real
            xfloat = pyfloat_asdouble(x)::Cdouble
            ans = tryconvert(T, TNumber, xfloat)
            if ans !== nothing
                return ans
            end
        elseif t.is_abstract_complex
            xcomplex = pycomplex_ascomplex(x)::Complex{Cdouble}
            ans = tryconvert(T, TNumber, xcomplex)
            if ans !== nothing
                return ans
            end
        end
    end

    # str -> AbstractString / AbstractChar / Symbol
    if (TAbstractString != Union{} || TAbstractChar != Union{} || Symbol <: T) && t.is_str
        xstr = pystr_asstring(x)
        # str -> AbstractString
        if TAbstractString != Union{}
            ans = tryconvert(T, TAbstractString, xstr)
            if ans !== nothing
                return ans
            end
        end
        # str -> Symbol
        if Symbol <: T
            return Some{T}(Symbol(xstr))
        end
        # str -> AbstractChar
        if TAbstractChar != Union{} && length(xstr) == 1
            xchar = xstr[1]::Char
            ans = tryconvert(T, TAbstractChar, xchar)
            if ans !== nothing
                return ans
            end
        end
    end

    # any -> Py
    if Py <: T
        return Some{T}(x)
    end
    nothing
end

function pytryconvert(::Type{T}, x::Any) where {T}
    return pytryconvert(T, Py(x)::Py)
end
export pytryconvert

Base.@kwdef struct TypeInfo
    type::Py
    # stdlib concrete types
    is_none::Bool = pyissubclass(type, pytype(pybuiltins.None))
    is_bool::Bool = pyissubclass(type, pybuiltins.bool)
    is_int::Bool = pyissubclass(type, pybuiltins.int)
    is_float::Bool = pyissubclass(type, pybuiltins.float)
    is_complex::Bool = pyissubclass(type, pybuiltins.complex)
    is_str::Bool = pyissubclass(type, pybuiltins.str)
    is_bytes::Bool = pyissubclass(type, pybuiltins.bytes)
    is_bytearray::Bool = pyissubclass(type, pybuiltins.bytearray)
    is_list::Bool = pyissubclass(type, pybuiltins.list)
    is_tuple::Bool = pyissubclass(type, pybuiltins.tuple)
    is_dict::Bool = pyissubclass(type, pybuiltins.dict)
    is_range::Bool = pyissubclass(type, pybuiltins.range)
    is_date::Bool = pyissubclass(type, pyimport("datetime").date)
    is_time::Bool = pyissubclass(type, pyimport("datetime").time)
    is_datetime::Bool = pyissubclass(type, pyimport("datetime").datetime)
    is_timedelta::Bool = pyissubclass(type, pyimport("datetime").timedelta)
    is_exception::Bool = pyissubclass(type, pybuiltins.BaseException)
    is_io::Bool = pyissubclass(type, pyimport("io").IOBase)
    # stdlib abstract types
    is_abstract_iterable::Bool = pyissubclass(type, pyimport("collections.abc").Iterable)
    is_abstract_collection::Bool = pyissubclass(type, pyimport("collections.abc").Collection)
    is_abstract_sequence::Bool = pyissubclass(type, pyimport("collections.abc").Sequence)
    is_abstract_set::Bool = pyissubclass(type, pyimport("collections.abc").Set)
    is_abstract_mapping::Bool = pyissubclass(type, pyimport("collections.abc").Mapping)
    is_abstract_number::Bool = pyissubclass(type, pyimport("numbers").Number)
    is_abstract_complex::Bool = pyissubclass(type, pyimport("numbers").Complex)
    is_abstract_real::Bool = pyissubclass(type, pyimport("numbers").Real)
    is_abstract_rational::Bool = pyissubclass(type, pyimport("numbers").Rational)
    is_abstract_integral::Bool = pyissubclass(type, pyimport("numbers").Integral)
    # arrays
    has_numpy_array_conversion::Bool = pyhasattr(type, "__array__")
    has_numpy_array_interface::Bool = pyhasattr(type, "__array_interface__")
    has_numpy_array_struct::Bool = pyhasattr(type, "__array_struct__")
    has_buffer_protocol::Bool = C.PyType_CheckBuffer(getptr(type))
    is_array_like::Bool = has_numpy_array_conversion || has_numpy_array_interface || has_numpy_array_struct || has_buffer_protocol
    # numpy (TODO)
    # ctypes (TODO)
end

const TYPEINFO_CACHE = Dict{Py,TypeInfo}()

typeinfo(t::Py) = get!(() -> TypeInfo(type=t), TYPEINFO_CACHE, t)

function tryconvert(::Type{T}, ::Type{S}, x) where {T,S<:T}
    try
        Some{T}(convert(S, x)::S)
    catch
        # TODO: only catch some exception types?
        nothing
    end
end

tryconvert(::Type{T}, ::Type{T}, x::T) where {T} = Some{T}(x)
tryconvert(::Type{T}, ::Type{S}, x::S) where {T,S<:T} = Some{T}(x)
