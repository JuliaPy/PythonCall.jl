"""
    pyimport(m, ...)
    pyimport(m => k, ...)

Import and return the module `m`.

If additionally `k` is given, then instead returns this attribute from `m`. If it is a tuple, a tuple of attributes is returned.

If two or more arguments are given, they are all imported and returned as a tuple.
"""
pyimport(m::AbstractString) = check(C.PyImport_ImportModule(m))
pyimport(m) = check(C.PyImport_Import(pyobject(m)))
function pyimport(x::Pair)
    m = pyimport(x[1])
    x[2] isa Tuple ? map(k->pygetattr(m, k), x[2]) : pygetattr(m, x[2])
end
pyimport(m1, m2, ms...) = (pyimport(m1), pyimport(m2), map(pyimport, ms)...)
export pyimport
