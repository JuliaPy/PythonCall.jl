const pyjlcollectiontype = pynew()

function init_collection()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class JlCollection(JlBase, _JlReprMixin):
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
                return self._jl_callmethod($(pyjl_methodnum(pyjlmixin_eq_bool)), other)
            else:
                return NotImplemented
        def copy(self):
            return type(self)(Base.copy(self))
    import collections.abc
    collections.abc.Collection.register(JlCollection)
    del collections
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlcollectiontype, jl.JlCollection)
end

"""
    pyjlcollection(x::Union{AbstractArray,AbstractSet,AbstractDict})

Wrap `x` as a Python `collections.abc.Collection` object.
"""
pyjlcollection(x::Union{AbstractArray,AbstractSet,AbstractDict}) = pyjl(pyjlcollectiontype, x)
export pyjlcollection

# Py(x::AbstractSet) = pyjlcollection(x)
