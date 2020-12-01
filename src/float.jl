const pyfloattype = pylazyobject(() -> pybuiltins.float)
export pyfloattype

pyfloat(args...; opts...) = pyfloattype(args...; opts...)
pyfloat(x::Real) = check(C.PyFloat_FromDouble(x))
export pyfloat

pyisfloat(o::PyObject) = pytypecheck(o, pyfloattype)
export pyisfloat

function pyfloat_tryconvert(::Type{T}, o::PyObject) where {T}
    x = check(C.PyFloat_AsDouble(o), true)
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
