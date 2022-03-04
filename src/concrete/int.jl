# :PyLong_FromLongLong => (Clonglong,) => PyPtr,
# :PyLong_FromUnsignedLongLong => (Culonglong,) => PyPtr,
# :PyLong_FromString => (Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint) => PyPtr,
# :PyLong_AsLongLong => (PyPtr,) => Clonglong,
# :PyLong_AsUnsignedLongLong => (PyPtr,) => Culonglong,

pyint_fallback(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) =
    pynew(errcheck(C.PyLong_FromString(string(x, base=32), C_NULL, 32)))
pyint_fallback(x::Integer) = pyint_fallback(BigInt(x))

"""
    pyint(x=0)

Convert `x` to a Python `int`.
"""
function pyint(x::Integer=0)
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
pyint(x) = @autopy x pynew(errcheck(C.PyNumber_Long(getptr(x_))))
export pyint

pyisint(x) = pytypecheckfast(x, C.Py_TPFLAGS_LONG_SUBCLASS)

pyconvert_rule_int(::Type{T}, x::Py) where {T<:Number} = begin
    # first try to convert to Clonglong (or Culonglong if unsigned)
    v = T <: Unsigned ? C.PyLong_AsUnsignedLongLong(getptr(x)) : C.PyLong_AsLongLong(getptr(x))
    if !iserrset_ambig(v)
        # success
        pydel!(x)
        return pyconvert_tryconvert(T, v)
    elseif errmatches(pybuiltins.OverflowError)
        # overflows Clonglong or Culonglong
        errclear()
        if T in (
               Bool,
               Int8,
               Int16,
               Int32,
               Int64,
               Int128,
               UInt8,
               UInt16,
               UInt32,
               UInt64,
               UInt128,
           ) &&
           typemin(typeof(v)) ≤ typemin(T) &&
           typemax(T) ≤ typemax(typeof(v))
            # definitely overflows S, give up now
            pydel!(x)
            return pyconvert_unconverted()
        else
            # try converting -> int -> str -> BigInt -> T
            x_int = pyint(x)
            pydel!(x)
            x_str = pystr(String, x_int)
            pydel!(x_int)
            v = parse(BigInt, x_str)
            return pyconvert_tryconvert(T, v)
        end
    else
        # other error
        pydel!(x)
        pythrow()
    end
end
