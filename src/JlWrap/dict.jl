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

pyjldict_contains(x::AbstractDict, k::Py) =
    Py(haskey(x, @pyconvert(keytype(x), k, return Py(false))))

pyjldict_getitem(x::AbstractDict, k::Py) = Py(x[pyconvert(keytype(x), k)])

pyjldict_setitem(x::AbstractDict, k::Py, v::Py) =
    (x[pyconvertarg(keytype(x), k, "key")] = pyconvertarg(valtype(x), v, "value");
    Py(nothing))

pyjldict_delitem(x::AbstractDict, k::Py) =
    (delete!(x, pyconvert(keytype(x), k)); Py(nothing))

function pyjldict_update(x::AbstractDict, items_::Py)
    for item_ in items_
        (k, v) = pyconvert(Tuple{keytype(x),valtype(x)}, item_)
        x[k] = v
    end
    Py(nothing)
end

const pyjldicttype = pynew()

function init_dict()
    jl = pyjuliacallmodule
    m = pyimport("juliacall._jlwrap.dict")
    methods = m._jl_methods
    methods["__iter__"] = pyjl_methodnum(pyjliter ∘ Iterator ∘ keys)
    methods["__contains__"] = pyjl_methodnum(pyjldict_contains)
    methods["__getitem__"] = pyjl_methodnum(pyjldict_getitem)
    methods["__setitem__"] = pyjl_methodnum(pyjldict_setitem)
    methods["__delitem__"] = pyjl_methodnum(pyjldict_delitem)
    methods["keys"] = pyjl_methodnum(pyjlset ∘ keys)
    methods["values"] = pyjl_methodnum(pyjlcollection ∘ values)
    methods["items"] = pyjl_methodnum(pyjlset ∘ DictPairSet)
    methods["popitem"] = pyjl_methodnum(Py ∘ pop!)
    methods["update"] = pyjl_methodnum(pyjldict_update)

    pycopy!(pyjldicttype, jl.JlDict)
end

"""
    pyjldict(x::AbstractDict)

Wrap `x` as a Python `dict`-like object.
"""
pyjldict(x::AbstractDict) = pyjl(pyjldicttype, x)

Py(x::AbstractDict) = pyjldict(x)
