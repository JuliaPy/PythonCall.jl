const pyjlsettype = pynew()

pyjlset_add(x::AbstractSet, v::Py) = (push!(x, pyconvertarg(eltype(x), v, "value")); Py(nothing))

function pyjlset_discard(x::AbstractSet, v_::Py)
    v = @pyconvert(eltype(x), v_, return Py(nothing))
    delete!(x, v)
    Py(nothing)
end

pyjlset_clear(x::AbstractSet) = (empty!(x); Py(nothing))

function pyjlset_pop(x::AbstractSet)
    if isempty(x)
        errset(pybuiltins.KeyError, "pop from an empty set")
        PyNULL
    else
        Py(pop!(x))
    end
end

function pyjlset_remove(x::AbstractSet, v_::Py)
    v = @pyconvert eltype(x) v_ begin
        errset(pybuiltins.KeyError, v_)
        return PyNULL
    end
    if v in x
        delete!(x, v)
        return Py(nothing)
    else
        errset(pybuiltins.KeyError, v_)
        return PyNULL
    end
end

function pyjlset_update(x::AbstractSet, vs_::Py)
    for v_ in vs_
        v = pyconvert(eltype(x), v_)
        push!(x, v)
        pydel!(v_)
    end
    Py(nothing)
end

function pyjlset_difference_update(x::AbstractSet, vs_::Py)
    for v_ in vs_
        v = @pyconvert(eltype(x), v_, continue)
        delete!(x, v)
        pydel!(v_)
    end
    Py(nothing)
end

function pyjlset_intersection_update(x::AbstractSet, vs_::Py)
    vs = Set{eltype(x)}()
    for v_ in vs_
        v = @pyconvert(eltype(x), v_, continue)
        push!(vs, v)
        pydel!(v_)
    end
    intersect!(x, vs)
    Py(nothing)
end

function pyjlset_symmetric_difference_update(x::AbstractSet, vs_::Py)
    vs = Set{eltype(x)}()
    for v_ in vs_
        v = pyconvert(eltype(x), v_)
        push!(vs, v)
        pydel!(v_)
    end
    symdiff!(x, vs)
    Py(nothing)
end

function init_jlwrap_set()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class SetValue(AnyValue):
        __slots__ = ()
        def add(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_add)), value)
        def discard(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_discard)), value)
        def clear(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_clear)))
        def copy(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ copy)))
        def pop(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_pop)))
        def remove(self, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_remove)), value)
        def difference(self, other):
            return set(self).difference(other)
        def intersection(self, other):
            return set(self).intersection(other)
        def symmetric_difference(self, other):
            return set(self).symmetric_difference(other)
        def union(self, other):
            return set(self).union(other)
        def isdisjoint(self, other):
            return set(self).isdisjoint(other)
        def issubset(self, other):
            return set(self).issubset(other)
        def issuperset(self, other):
            return set(self).issuperset(other)
        def difference_update(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_difference_update)), other)
        def intersection_update(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_intersection_update)), other)
        def symmetric_difference_update(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_symmetric_difference_update)), other)
        def update(self, other):
            return self._jl_callmethod($(pyjl_methodnum(pyjlset_update)), other)
    import collections.abc
    collections.abc.MutableSet.register(SetValue)
    del collections
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjlsettype, jl.SetValue)
end

pyjltype(::AbstractSet) = pyjlsettype
