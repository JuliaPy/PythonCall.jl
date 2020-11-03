const pytupletype = PyLazyObject(() -> pybuiltins.tuple)
export pytupletype

pyistuple(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_TUPLE_SUBCLASS)
export pyistuple

pytuple() = cpycall_obj(Val(:PyTuple_New), CPy_ssize_t(0))
pytuple(args...; opts...) = pytupletype(args...; opts...)
pytuple(x::Union{Tuple,AbstractVector}) = pytuple_fromiter(x)
export pytuple

function pytuple_fromiter(xs)
    n = length(xs)
    t = cpycall_obj(Val(:PyTuple_New), CPy_ssize_t(n))
    for (i,x) in enumerate(xs)
        xo = pyobject(x)
        cpycall_void(Val(:PyTuple_SetItem), t, CPy_ssize_t(i-1), pyincref!(xo))
    end
    return t
end
