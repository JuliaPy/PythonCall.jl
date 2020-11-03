const pybooltype = PyLazyObject(() -> pybuiltins.bool)
export pybooltype

const pytrue = PyLazyObject(() -> pybuiltins.True)
const pyfalse = PyLazyObject(() -> pybuiltins.False)
export pytrue, pyfalse

pybool(args...; opts...) = pybooltype(args...; opts...)
pybool(o::Bool) = o ? pytrue : pyfalse
export pybool

pyisbool(o::AbstractPyObject) = pytypecheck(o, pybooltype)
export pyisbool

function pybool_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    x = pytruth(o)
    if Bool <: T
        pytruth(o)
    else
        tryconvert(T, x)
    end
end
