const pylisttype = PyLazyObject(() -> pybuiltins.list)
export pylisttype

pyislist(o::AbstractPyObject) = pytypecheckfast(o, C.Py_TPFLAGS_LIST_SUBCLASS)
export pyislist

pylist() = check(C.PyList_New(0))
pylist(args...; opts...) = pylisttype(args...; opts...)
pylist(x::Union{Tuple,AbstractVector}) = pylist_fromiter(x)
export pylist

function pylist_fromiter(xs)
    r = pylist()
    for x in xs
        xo = pyobject(x)
        check(C.PyList_Append(r, xo))
    end
    return r
end

pycollist(x::AbstractArray{T,N}) where {T,N} = N==0 ? pyobject(x[]) : pylist_fromiter(pycollist(y) for y in eachslice(x; dims=N))
export pycollist

pyrowlist(x::AbstractArray{T,N}) where {T,N} = N==0 ? pyobject(x[]) : pylist_fromiter(pyrowlist(y) for y in eachslice(x; dims=1))
export pyrowlist
