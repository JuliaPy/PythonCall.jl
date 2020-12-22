# const pypandas = pylazyobject(() -> pyimport("pandas"))
# const pypandasdataframetype = pylazyobject(() -> pypandas.DataFrame)

# asvector(x::AbstractVector) = x
# asvector(x) = collect(x)

# """
#     pycolumntable(src)

# Construct a "column table" from the `Tables.jl`-compatible table `src`, namely a Python `dict` mapping column names to column vectors.
# """
# function pycolumntable(src)
#     cols = Tables.columns(src)
#     pydict_fromiter(pystr(String(n)) => asvector(Tables.getcolumn(cols, n)) for n in Tables.columnnames(cols))
# end
# pycolumntable(; cols...) = pycolumntable(cols)
# export pycolumntable

# """
#     pyrowtable(src)

# Construct a "row table" from the `Tables.jl`-compatible table `src`, namely a Python `list` of rows, each row being a Python `dict` mapping column names to values.
# """
# function pyrowtable(src)
#     rows = Tables.rows(src)
#     names = Tables.columnnames(rows)
#     pynames = [pystr(String(n)) for n in names]
#     pylist_fromiter(pydict_fromiter(pn => Tables.getcolumn(row, n) for (n,pn) in zip(names, pynames)) for row in rows)
# end
# pyrowtable(; cols...) = pyrowtable(cols)
# export pyrowtable

# """
#     pypandasdataframe([src]; ...)

# Construct a pandas dataframe from `src`.

# Usually equivalent to `pyimport("pandas").DataFrame(src, ...)`, but `src` may also be `Tables.jl`-compatible table.
# """
# pypandasdataframe(t::PyObject; opts...) = pypandasdataframetype(t; opts...)
# pypandasdataframe(; opts...) = pypandasdataframetype(; opts...)
# function pypandasdataframe(t; opts...)
#     if Tables.istable(t)
#         cs = Tables.columns(t)
#         pypandasdataframetype(pydict_fromstringiter(string(c) => asvector(Tables.getcolumn(cs, c)) for c in Tables.columnnames(cs)); opts...)
#     else
#         pypandasdataframetype(t; opts...)
#     end
# end
# export pypandasdataframe

multidict(src) = Dict(k=>v for (ks,v) in src for k in (ks isa Vector ? ks : [ks]))

"""
    PyPandasDataFrame(o; indexname=:index, columntypes=(), copy=false)

Wrap the Pandas dataframe `o` as a Julia table.

This object satisfies the `Tables.jl` and `TableTraits.jl` interfaces.

- `:indexname` is the name of the index column when converting this to a table, and may be `nothing` to exclude the index.
- `:columntypes` is an iterable of `columnname=>type` or `[columnnames...]=>type` pairs, used when converting to a table.
- `:copy` is true to copy columns on conversion.
"""
struct PyPandasDataFrame
    ref :: PyRef
    indexname :: Union{Symbol, Nothing}
    columntypes :: Dict{Symbol, Type}
    copy :: Bool
end
PyPandasDataFrame(o; indexname=:index, columntypes=(), copy=false) = PyPandasDataFrame(PyRef(o), indexname, multidict(columntypes), copy)
export PyPandasDataFrame

ispyreftype(::Type{PyPandasDataFrame}) = true
pyptr(df::PyPandasDataFrame) = df.ref
Base.unsafe_convert(::Type{CPyPtr}, df::PyPandasDataFrame) = checknull(pyptr(df))
C.PyObject_TryConvert__initial(o, ::Type{PyPandasDataFrame}) = C.putresult(PyPandasDataFrame(pyborrowedref(o)))

Base.show(io::IO, x::PyPandasDataFrame) = print(io, pystr(String, x))
Base.show(io::IO, mime::MIME, o::PyPandasDataFrame) = _py_mime_show(io, mime, o)
Base.showable(mime::MIME, o::PyPandasDataFrame) = _py_mime_showable(mime, o)

### Tables.jl / TableTraits.jl integration

Tables.istable(::Type{PyPandasDataFrame}) = true
Tables.columnaccess(::Type{PyPandasDataFrame}) = true
function Tables.columns(x::PyPandasDataFrame)
    error("not implemented")
    # # columns
    # names = Symbol[Symbol(pystr(String, c)) for c in x.o.columns]
    # columns = PyVector[PyVector{get(x.columntypes, c, missing)}(x.o[pystr(c)].to_numpy()) for c in names]
    # # index
    # if x.indexname !== nothing
    #     if x.indexname ∈ names
    #         error("table already has column called $(x.indexname), cannot use it for index")
    #     else
    #         pushfirst!(names, x.indexname)
    #         pushfirst!(columns, PyArray{get(x.columntypes, x.indexname, missing)}(x.o.index.to_numpy()))
    #     end
    # end
    # if x.copy
    #     columns = map(copy, columns)
    # end
    # return NamedTuple{Tuple(names)}(Tuple(columns))
end

IteratorInterfaceExtensions.isiterable(x::PyPandasDataFrame) = true
IteratorInterfaceExtensions.getiterator(x::PyPandasDataFrame) = IteratorInterfaceExtensions.getiterator(Tables.rows(x))

TableTraits.isiterabletable(x::PyPandasDataFrame) = true
