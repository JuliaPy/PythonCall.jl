const pyjlcollectiontype = pynew()

pyjlcollection_clear(x) = (empty!(x); Py(nothing))

pyjlcollection_contains(x, v::Py) =
    pybool(in(@pyconvert(eltype(x), v, (return Py(false))), x)::Bool)

pyjlcollection_eq(self, other) = pybool((self == pyjlvalue(other))::Bool)

function init_collection()
    jl = pyjuliacallmodule
    pybuiltins.exec(
        pybuiltins.compile(
            """
$("\n"^(@__LINE__()-1))
class JlCollection(JlBase2):
    __slots__ = ()
    def __len__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyint ∘ length)))
    def __bool__(self):
        return self._jl_callmethod($(pyjl_methodnum(pybool ∘ !isempty)))
    def __iter__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjliter ∘ Iterator)))
    def __hash__(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlany_hash)))
    def __eq__(self, other):
        if isinstance(self, type(other)) or isinstance(other, type(self)):
            return self._jl_callmethod($(pyjl_methodnum(pyjlcollection_eq)), other)
        else:
            return NotImplemented
    def __contains__(self, v):
        return self._jl_callmethod($(pyjl_methodnum(pyjlcollection_contains)), v)
    def copy(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlcollection ∘ copy)))
    def clear(self):
        return self._jl_callmethod($(pyjl_methodnum(pyjlcollection_clear)))
import collections.abc
collections.abc.Collection.register(JlCollection)
del collections
""",
            @__FILE__(),
            "exec",
        ),
        jl.__dict__,
    )
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
