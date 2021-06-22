# :PyLong_FromLongLong => (Clonglong,) => PyPtr,
# :PyLong_FromUnsignedLongLong => (Culonglong,) => PyPtr,
# :PyLong_FromString => (Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint) => PyPtr,
# :PyLong_AsLongLong => (PyPtr,) => Clonglong,
# :PyLong_AsUnsignedLongLong => (PyPtr,) => Culonglong,

pyint_fallback(x::Union{Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt}) =
    pynew(errcheck(C.PyLong_FromString(string(x, base=32), C_NULL, 32)))
pyint_fallback(x::Integer) = pyint_fallback(BigInt(x))

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
pyint(x) = ispy(x) ? pynew(errcheck(C.PyNumber_Long(getptr(x)))) : pyint(convert(Integer, x))
export pyint
