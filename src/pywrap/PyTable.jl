"""
    PyTable(x)

Wrap `x` as a Tables.jl-compatible table.

`PyTable` is an abstract type. See [`PyPandasDataFrame`](@ref) for a concrete example.
"""
abstract type PyTable end
export PyTable

PyTable(x) = pyconvert(PyTable, x)

# Tables.istable(x::Py) = Tables.istable(@pyconvert(PyTable, x, return false))
# Tables.rowaccess(x::Py) = Tables.rowaccess(@pyconvert(PyTable, x, return false))
# Tables.rows(x::Py) = Tables.rows(pyconvert(PyTable, x))
# Tables.columnaccess(x::Py) = Tables.columnaccess(@pyconvert(PyTable, x, return false))
# Tables.columns(x::Py) = Tables.columns(pyconvert(PyTable, x))
# Tables.materializer(x::Py) = Tables.materializer(pyconvert(PyTable, x))
