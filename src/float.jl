const pyfloattype = PyLazyObject(() -> pybuiltins.float)
export pyfloattype

pyfloat(args...; opts...) = pyfloattype(args...; opts...)
pyfloat(x::Real) = cpycall_obj(Val(:PyFloat_FromDouble), convert(Cdouble, x))
export pyfloat

pyisfloat(o::AbstractPyObject) = pytypecheck(o, pyfloattype)
export pyisfloat

function pyfloat_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    x = cpycall_num_ambig(Val(:PyFloat_AsDouble), Cdouble, o)
    if (S = _typeintersect(T, Cdouble)) != Union{}
        convert(S, x)
    elseif (S = _typeintersect(T, AbstractFloat)) != Union{}
        convert(S, x)
    elseif (S = _typeintersect(T, Real)) != Union{}
        convert(S, x)
    else
        tryconvert(T, x)
    end
end
