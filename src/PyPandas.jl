const pypandas = PyLazyObject(() -> pyimport("pandas"))
const pypandasdataframetype = PyLazyObject(() -> pypandas.DataFrame)

asvector(x::AbstractVector) = x
asvector(x) = collect(x)

"""
    pypandasdataframe([src]; ...)

Construct a pandas dataframe from `src`.

Usually equivalent to `pyimport("pandas").DataFrame(src, ...)`, but `src` may also be `Tables.jl`-compatible table.
"""
pypandasdataframe(t::AbstractPyObject; opts...) = pypandasdataframetype(t; opts...)
pypandasdataframe(; opts...) = pypandasdataframetype(; opts...)
function pypandasdataframe(t; opts...)
    if Tables.istable(t)
        cs = Tables.columns(t)
        pypandasdataframetype(pydict_fromstringiter(string(c) => asvector(Tables.getcolumn(cs, c)) for c in Tables.columnnames(cs)); opts...)
    else
        pypandasdataframetype(t; opts...)
    end
end
export pypandasdataframe

"""
    PyPandasDataFrame(o; indexname=:index, columntypes=Dict(), copy=false)

Wrap the Pandas dataframe `o` as a Julia table.

This object satisfies the `Tables.jl` and `TableTraits.jl` interfaces.

- `:indexname` is the name of the index column when converting this to a table, and may be `nothing` to exclude the index.
- `:columntypes` is a dictionary mapping column names to types, mainly used to convert columns of type `object` to a corresponding Julia type.
- `:copy` is true to copy columns on conversion.
"""
struct PyPandasDataFrame
    o :: PyObject
    indexname :: Union{Symbol, Nothing}
    columntypes :: Dict{Symbol, Type}
    copy :: Bool
end
PyPandasDataFrame(o::AbstractPyObject; indexname=:index, columntypes=Dict(), copy=false) = PyPandasDataFrame(o, indexname, columntypes, copy)
export PyPandasDataFrame

function pypandasdataframe_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if PyPandasDataFrame <: T
        return PyPandasDataFrame(o)
    else
        tryconvert(T, PyPandasDataFrame(o))
    end
end

Base.show(io::IO, x::PyPandasDataFrame) = print(io, x.o)

### Tables.jl / TableTraits.jl integration

Tables.istable(::Type{PyPandasDataFrame}) = true
Tables.columnaccess(::Type{PyPandasDataFrame}) = true
function Tables.columns(x::PyPandasDataFrame)
    # columns
    names = Symbol[Symbol(pystr(String, c)) for c in x.o.columns]
    columns = PyVector[PyVector{get(x.columntypes, c, missing)}(x.o[pystr(c)].to_numpy()) for c in names]
    # index
    if x.indexname !== nothing
        if x.indexname ∈ names
            error("table already has column called $(x.indexname), cannot use it for index")
        else
            pushfirst!(names, x.indexname)
            pushfirst!(columns, PyArray{get(x.columntypes, x.indexname, missing)}(x.o.index.to_numpy()))
        end
    end
    if x.copy
        columns = map(copy, columns)
    end
    return NamedTuple{Tuple(names)}(Tuple(columns))
end

IteratorInterfaceExtensions.isiterable(x::PyPandasDataFrame) = true
IteratorInterfaceExtensions.getiterator(x::PyPandasDataFrame) = IteratorInterfaceExtensions.getiterator(Tables.rows(x))

TableTraits.isiterabletable(x::PyPandasDataFrame) = true
