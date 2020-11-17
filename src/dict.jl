const pydicttype = PyLazyObject(() -> pybuiltins.dict)
export pydicttype

pyisdict(o::AbstractPyObject) = pytypecheckfast(o, C.Py_TPFLAGS_DICT_SUBCLASS)
export pyisdict

pydict(args...; opts...) = pydicttype(args...; opts...)
pydict() = check(C.PyDict_New())
pydict(o::Union{AbstractDict, Base.Iterators.Pairs, AbstractArray{<:Pair}}) = pydict_fromiter(o)
pydict(o::NamedTuple) = pydict_fromiter(pairs(o))
export pydict

function pydict_fromiter(kvs)
    d = pydict()
    for (k,v) in kvs
        vo = pyobject(v)
        if k isa AbstractString
            check(C.PyDict_SetItemString(d, k, vo))
        else
            check(C.PyDict_SetItem(d, pyobject(k), vo))
        end
    end
    return d
end

function pydict_fromstringiter(kvs)
    d = pydict()
    for (k,v) in kvs
        ko = k isa AbstractString ? k : k isa Symbol ? String(k) : error("only string and symbols allowed")
        vo = pyobject(v)
        check(C.PyDict_SetItemString(d, ko, vo))
    end
    return d
end
