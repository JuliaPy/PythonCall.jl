const pytupletype = PyLazyObject(() -> pybuiltins.tuple)
export pytupletype

pyistuple(o::AbstractPyObject) = pytypecheckfast(o, C.Py_TPFLAGS_TUPLE_SUBCLASS)
export pyistuple

pytuple() = check(C.PyTuple_New(0))
pytuple(args...; opts...) = pytupletype(args...; opts...)
pytuple(x::Union{Tuple,AbstractVector,Pair}) = pytuple_fromiter(x)
export pytuple

function pytuple_fromiter(xs)
    n = length(xs)
    t = check(C.PyTuple_New(n))
    for (i,x) in enumerate(xs)
        xo = pyobject(x)
        check(C.PyTuple_SetItem(t, i-1, pyincref!(xo)))
    end
    return t
end

function pytuple_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if (S = _typeintersect(T, Tuple)) != Union{}
        pyiterable_tryconvert(S, o)
    else
        PyConvertFail()
    end
end
