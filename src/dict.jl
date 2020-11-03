const pydicttype = PyLazyObject(() -> pybuiltins.dict)
export pydicttype

pyisdict(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_DICT_SUBCLASS)
export pyisdict

pydict(args...; opts...) = pydicttype(args...; opts...)
pydict() = cpycall_obj(Val(:PyDict_New))
pydict(o::Union{AbstractDict, NamedTuple, Base.Iterators.Pairs, AbstractArray{<:Pair}}) = pydict_fromiter(o)
export pydict

function pydict_fromiter(kvs)
    d = pydict()
    for (k,v) in kvs
        vo = pyobject(v)
        if k isa AbstractString
            cpycall_void(Val(:PyDict_SetItemString), d, k, vo)
        else
            cpycall_void(Val(:PyDict_SetItem), d, pyobject(k), vo)
        end
    end
    return d
end

function pydict_fromstringiter(kvs)
    d = pydict()
    for (k,v) in kvs
        ko = k isa AbstractString ? k : k isa Symbol ? String(k) : error("only string and symbols allowed")
        vo = pyobject(v)
        cpycall_void(Val(:PyDict_SetItemString), d, ko, vo)
    end
    return d
end
