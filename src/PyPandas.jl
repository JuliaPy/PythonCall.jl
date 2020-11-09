struct PyPandasDataFrame
    o :: PyObject
end
export PyPandasDataFrame

function pypandasdataframe_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if PyPandasDataFrame <: T
        return PyPandasDataFrame(o)
    else
        tryconvert(T, PyPandasDataFrame(o))
    end
end

### Tables.jl integration

Tables.istable(::Type{PyPandasDataFrame}) = true
Tables.columnaccess(::Type{PyPandasDataFrame}) = true
function Tables.columns(x::PyPandasDataFrame)
    # columns
    names = Symbol[Symbol(pystr(String, c)) for c in x.o.columns]
    columns = PyArray[PyArray(x.o[pystr(c)].to_numpy()) for c in names]
    # index
    if :index ∉ names
        pushfirst!(names, :index)
        pushfirst!(columns, PyArray(x.o.index.to_numpy()))
    end
    return NamedTuple{Tuple(names)}(Tuple(columns))
end
# TODO: Tables.materializer
