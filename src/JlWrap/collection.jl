const pyjlcollectiontype = pynew()

pyjlcollection_clear(x) = (empty!(x); Py(nothing))

pyjlcollection_contains(x, v::Py) =
    pybool(in(@pyconvert(eltype(x), v, (return Py(false))), x)::Bool)

pyjlcollection_eq(self, other) = pybool((self == pyjlvalue(other))::Bool)

function init_collection()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.collection")
    methods = m._jl_methods
    methods["__len__"] = pyjl_methodnum(pyint ∘ length)
    methods["__bool__"] = pyjl_methodnum(pybool ∘ !isempty)
    methods["__iter__"] = pyjl_methodnum(pyjliter ∘ Iterator)
    methods["__hash__"] = pyjl_methodnum(pyjlany_hash)
    methods["__eq__"] = pyjl_methodnum(pyjlcollection_eq)
    methods["__contains__"] = pyjl_methodnum(pyjlcollection_contains)
    methods["copy"] = pyjl_methodnum(pyjlcollection ∘ copy)
    methods["clear"] = pyjl_methodnum(pyjlcollection_clear)

    pycopy!(pyjlcollectiontype, jl.JlCollection)
end

"""
    pyjlcollection(x)

Wrap `x` as a Python `collections.abc.Collection` object.

The argument should be a collection of values, in the sense of supporting `iterate`,
`hash`, `in` and `length`. This includes `AbstractArray`, `AbstractSet`, `AbstractDict`,
`Tuple`, `Base.RefValue` (`Ref(...)`) and `Base.ValueIterator` (`values(Dict(...))`).
"""
pyjlcollection(x) = pyjl(pyjlcollectiontype, x)
pyjlcollection(x::AbstractSet) = pyjlset(x)
pyjlcollection(x::AbstractArray) = pyjlarray(x)
pyjlcollection(x::AbstractDict) = pyjldict(x)

Py(x::Base.ValueIterator) = pyjlcollection(x)
Py(x::Base.RefValue) = pyjlcollection(x)
