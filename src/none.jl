const pynone = pylazyobject(() -> pybuiltins.None)
export pynone

const pynonetype = pylazyobject(() -> pytype(pynone))
export pynonetype

pyisnone(o::PyObject) = pyis(o, pynone)
export pyisnone

function pynone_tryconvert(::Type{T}, o::PyObject) where {T}
    tryconvert(T, nothing)
end
