"""
    pycollist([T=PyObject,] x::AbstractArray) :: T

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of columns.
"""
pycollist(::Type{T}, x::AbstractArray) where {T} =
    ndims(x) == 0 ? pyconvert(T, x[]) :
    pylist(T, pycollist(PyRef, y) for y in eachslice(x; dims = ndims(x)))
pycollist(x::AbstractArray) = pycollist(PyObject, x)
export pycollist

"""
    pyrowlist([T=PyObject,] x::AbstractArray) :: T

Create a nested Python `list`-of-`list`s from the elements of `x`. For matrices, this is a list of rows.
"""
pyrowlist(::Type{T}, x::AbstractArray) where {T} =
    ndims(x) == 0 ? pyconvert(T, x[]) :
    pylist(T, pyrowlist(PyRef, y) for y in eachslice(x; dims = 1))
pyrowlist(x::AbstractArray) = pyrowlist(PyObject, x)
export pyrowlist
