struct DictPairSet{K,V,T<:AbstractDict{K,V}} <: AbstractSet{Tuple{K,V}}
    dict::T
end
Base.length(x::DictPairSet) = length(x.dict)
Base.iterate(x::DictPairSet) =
    (r = iterate(x.dict); r === nothing ? nothing : (Tuple(r[1]), r[2]))
Base.iterate(x::DictPairSet, st) =
    (r = iterate(x.dict, st); r === nothing ? nothing : (Tuple(r[1]), r[2]))
Base.in(v::Pair, x::DictPairSet) = v in x.dict
Base.in(v::Tuple{Any,Any}, x::DictPairSet) = Pair(v[1], v[2]) in x.dict

pyjldict_iter(x::AbstractDict) = Py(Iterator(keys(x)))

pyjldict_contains(x::AbstractDict, k::Py) = Py(haskey(x, @pyconvert(keytype(x), k, return Py(false))))

pyjldict_clear(x::AbstractDict) = (empty!(x); Py(nothing))

pyjldict_getitem(x::AbstractDict, k::Py) = Py(x[pyconvert(keytype(x), k)])

pyjldict_setitem(x::AbstractDict, k::Py, v::Py) = (x[pyconvertarg(keytype(x), k, "key")] = pyconvertarg(valtype(x), v, "value"); Py(nothing))

pyjldict_delitem(x::AbstractDict, k::Py) = (delete!(x, pyconvert(keytype(x), k)); Py(nothing))

function pyjldict_update(x::AbstractDict, items_::Py)
    for item_ in items_
        (k, v) = pyconvert_and_del(Tuple{keytype(x), valtype(x)}, item_)
        x[k] = v
    end
    Py(nothing)
end

const pyjldicttype = pynew()

function init_jlwrap_dict()
    jl = pyjuliacallmodule
    pybuiltins.exec(pybuiltins.compile("""
    $("\n"^(@__LINE__()-1))
    class DictValue(AnyValue):
        __slots__ = ()
        __module__ = "juliacall"
        _jl_undefined_ = object()
        def __iter__(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjldict_iter)))
        def __contains__(self, key):
            return self._jl_callmethod($(pyjl_methodnum(pyjldict_contains)), key)
        def __getitem__(self, key):
            if key in self:
                return self._jl_callmethod($(pyjl_methodnum(pyjldict_getitem)), key)
            else:
                raise KeyError(key)
        def __setitem__(self, key, value):
            return self._jl_callmethod($(pyjl_methodnum(pyjldict_setitem)), key, value)
        def __delitem__(self, key):
            if key in self:
                return self._jl_callmethod($(pyjl_methodnum(pyjldict_delitem)), key)
            else:
                raise KeyError(key)
        def keys(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ keys)))
        def values(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ values)))
        def items(self):
            return self._jl_callmethod($(pyjl_methodnum(Py ∘ DictPairSet)))
        def get(self, key, default=None):
            if key in self:
                return self[key]
            else:
                return default
        def setdefault(self, key, default=None):
            if key not in self:
                self[key] = default
            return self[key]
        def clear(self):
            return self._jl_callmethod($(pyjl_methodnum(pyjldict_clear)))
        def pop(self, key, default=_jl_undefined_):
            if key in self:
                ans = self[key]
                del self[key]
                return ans
            elif default is self._jl_undefined_:
                raise KeyError(key)
            else:
                return default
        def popitem(self):
            if len(self):
                return self._jl_callmethod($(pyjl_methodnum(Py ∘ pop!)))
            else:
                raise KeyError()
        def update(self, other=_jl_undefined_, **kwargs):
            if other is self._jl_undefined_:
                pass
            else:
                if hasattr(other, "keys"):
                    items = ((k, other[k]) for k in other.keys())
                else:
                    items = other
                self._jl_callmethod($(pyjl_methodnum(pyjldict_update)), items)
            if kwargs:
                self.update(kwargs)
    import collections.abc
    collections.abc.MutableMapping.register(DictValue)
    del collections
    """, @__FILE__(), "exec"), jl.__dict__)
    pycopy!(pyjldicttype, jl.DictValue)
end

pyjltype(::AbstractDict) = pyjldicttype
