const pybooltype = pylazyobject(() -> pybuiltins.bool)
export pybooltype

const pytrue = pylazyobject(() -> pybuiltins.True)
const pyfalse = pylazyobject(() -> pybuiltins.False)
export pytrue, pyfalse

pybool(args...; opts...) = pybooltype(args...; opts...)
pybool(o::Bool) = o ? pytrue : pyfalse
export pybool

pyisbool(o::PyObject) = pytypecheck(o, pybooltype)
export pyisbool

function pybool_tryconvert(::Type{T}, o::PyObject) where {T}
    x = pytruth(o)
    if Bool <: T
        pytruth(o)
    else
        tryconvert(T, x)
    end
end
