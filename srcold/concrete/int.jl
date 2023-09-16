# :PyLong_FromLongLong => (Clonglong,) => PyPtr,
# :PyLong_FromUnsignedLongLong => (Culonglong,) => PyPtr,
# :PyLong_FromString => (Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint) => PyPtr,
# :PyLong_AsLongLong => (PyPtr,) => Clonglong,
# :PyLong_AsUnsignedLongLong => (PyPtr,) => Culonglong,

pyconvert_rule_int(::Type{T}, x::Py) where {T<:Number} = begin
    # first try to convert to Clonglong (or Culonglong if unsigned)
    v = T <: Unsigned ? C.PyLong_AsUnsignedLongLong(getptr(x)) : C.PyLong_AsLongLong(getptr(x))
    if !iserrset_ambig(v)
        # success
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
            return pyconvert_unconverted()
        else
            # try converting -> int -> str -> BigInt -> T
            x_int = pyint(x)
            x_str = pystr(String, x_int)
            pydel!(x_int)
            v = parse(BigInt, x_str)
            return pyconvert_tryconvert(T, v)
        end
    else
        # other error
        pythrow()
    end
end
