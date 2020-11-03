const pynone = PyLazyObject(() -> pybuiltins.None)
export pynone

const pynonetype = PyLazyObject(() -> pytype(pynone))
export pynonetype

pyisnone(o::AbstractPyObject) = pyis(o, pynone)
export pyisnone

function pynone_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    tryconvert(T, nothing)
end
