asvector(x::AbstractVector) = x
asvector(x) = collect(x)

"""
    pycolumntable([T=PyObject,] src) :: T

Construct a "column table" from the `Tables.jl`-compatible table `src`, namely a Python `dict` mapping column names to column vectors.
"""
function pycolumntable(::Type{T}, src) where {T}
    cols = Tables.columns(src)
    pydict(
        T,
        pystr(String(n)) => asvector(Tables.getcolumn(cols, n)) for
        n in Tables.columnnames(cols)
    )
end
pycolumntable(::Type{T}; cols...) where {T} = pycolumntable(T, cols)
pycolumntable(src) = pycolumntable(PyObject, src)
pycolumntable(; opts...) = pycolumntable(PyObject, opts)
export pycolumntable

"""
    pyrowtable([T=PyObject,] src) :: T

Construct a "row table" from the `Tables.jl`-compatible table `src`, namely a Python `list` of rows, each row being a Python `dict` mapping column names to values.
"""
function pyrowtable(::Type{T}, src) where {T}
    rows = Tables.rows(src)
    names = Tables.columnnames(rows)
    pynames = [pystr(String(n)) for n in names]
    pylist(
        T,
        pydict(pn => Tables.getcolumn(row, n) for (n, pn) in zip(names, pynames)) for
        row in rows
    )
end
pyrowtable(::Type{T}; cols...) where {T} = pyrowtable(T, cols)
pyrowtable(src) = pyrowtable(PyObject, src)
pyrowtable(; opts...) = pyrowtable(PyObject, opts)
export pyrowtable

aspandasvector(x) = asvector(x)

@init @require CategoricalArrays="324d7699-5711-5eae-9e2f-1d82baa6b597" @eval begin
    aspandasvector(x::CategoricalArrays.CategoricalArray) = begin
        codes = map(x -> x===missing ? -1 : Int(CategoricalArrays.levelcode(x))-1, x)
        cats = CategoricalArrays.levels(x)
        ordered = x.pool.ordered
        pypandasmodule().Categorical.from_codes(codes, cats, ordered=ordered)
    end
end

"""
    pypandasdataframe([T=PyObject,] [src]; ...) :: T

Construct a pandas dataframe from `src`.

Usually equivalent to `pyimport("pandas").DataFrame(src, ...)`, but `src` may also be `Tables.jl`-compatible table.
"""
pypandasdataframe(::Type{T}; opts...) where {T} = pycall(T, pypandasmodule().DataFrame; opts...)
pypandasdataframe(::Type{T}, t; opts...) where {T} = begin
    if Tables.istable(t)
        cs = Tables.columns(t)
        pycall(T, pypandasmodule().DataFrame, pydict(pystr(String(n)) => aspandasvector(Tables.getcolumn(cs, n)) for n in Tables.columnnames(cs)); opts...)
    else
        pycall(T, pypandasmodule().DataFrame, t; opts...)
    end
end
pypandasdataframe(args...; opts...) = pypandasdataframe(PyObject, args...; opts...)
export pypandasdataframe

multidict(src) = Dict(k => v for (ks, v) in src for k in (ks isa Vector ? ks : [ks]))

"""
    PyPandasDataFrame(o; indexname=:index, columntypes=(), copy=false)

Wrap the Pandas dataframe `o` as a Julia table.

This object satisfies the `Tables.jl` and `TableTraits.jl` interfaces.

- `indexname`: The name of the index column when converting this to a table, and may be `nothing` to exclude the index.
- `columntypes`: An iterable of `columnname=>type` or `[columnnames...]=>type` pairs, used when converting to a table.
- `copy`: True to copy columns on conversion.
"""
mutable struct PyPandasDataFrame
    ptr::CPyPtr
    indexname::Union{Symbol,Nothing}
    columntypes::Dict{Symbol,Type}
    copy::Bool
    PyPandasDataFrame(::Val{:new}, ptr::Ptr, indexname::Union{Symbol,Nothing}, columntypes::Dict{Symbol,Type}, copy::Bool) =
        finalizer(pyref_finalize!, new(CPyPtr(ptr), indexname, columntypes, copy))
end
PyPandasDataFrame(o; indexname::Union{Symbol,Type} = :index, columntypes = (), copy::Bool = false) =
    PyPandasDataFrame(Val(:new), checknull(C.PyObject_From(o)), indexname, multidict(columntypes), copy)
export PyPandasDataFrame

ispyreftype(::Type{PyPandasDataFrame}) = true
pyptr(df::PyPandasDataFrame) = df.ptr
Base.unsafe_convert(::Type{CPyPtr}, df::PyPandasDataFrame) = checknull(pyptr(df))
C.PyObject_TryConvert__initial(o, ::Type{PyPandasDataFrame}) =
    C.putresult(PyPandasDataFrame(pyborrowedref(o)))

Base.show(io::IO, x::PyPandasDataFrame) = print(io, pystr(String, x))
Base.show(io::IO, mime::MIME, o::PyPandasDataFrame) = _py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/plain", o::PyPandasDataFrame) = _py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/csv", o::PyPandasDataFrame) = _py_mime_show(io, mime, o)
Base.show(io::IO, mime::MIME"text/tab-separated-values", o::PyPandasDataFrame) = _py_mime_show(io, mime, o)
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
IteratorInterfaceExtensions.getiterator(x::PyPandasDataFrame) =
    IteratorInterfaceExtensions.getiterator(Tables.rows(x))

TableTraits.isiterabletable(x::PyPandasDataFrame) = true
