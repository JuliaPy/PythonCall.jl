const pyslicetype = PyLazyObject(() -> pybuiltins.slice)
export pyslicetype

pyslice(args...; opts...) = pyslicetype(args...; opts...)
export pyslice

pyisslice(o::AbstractPyObject) = pytypecheck(o, pyslicetype)
export pyisslice
