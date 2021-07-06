"""
    PyPandasDataFrame(x)

Wraps the pandas DataFrame `x` as a Tables.jl-compatible table.
"""
struct PyPandasDataFrame <: PyTable
    py::Py
    PyPandasDataFrame(::Val{:new}, py::Py) = new(py)
end
export PyPandasDataFrame

PyPandasDataFrame(x) = PyPandasDataFrame(Val(:new), Py(x))

ispy(x::PyPandasDataFrame) = true
getpy(x::PyPandasDataFrame) = x.py
pydel!(x::PyPandasDataFrame) = pydel!(x.py)

pyconvert_rule_pandasdataframe(::Type{PyPandasDataFrame}, x::Py) = pyconvert_return(PyPandasDataFrame(x))

Tables.istable(::Type{PyPandasDataFrame}) = true
Tables.columnaccess(::Type{PyPandasDataFrame}) = true
