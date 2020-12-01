const pyslicetype = pylazyobject(() -> pybuiltins.slice)
export pyslicetype

pyslice(args...; opts...) = pyslicetype(args...; opts...)
export pyslice

pyisslice(o::PyObject) = pytypecheck(o, pyslicetype)
export pyisslice
