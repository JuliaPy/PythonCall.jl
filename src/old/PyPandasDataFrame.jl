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

multidict(src) = Dict{String,Type}(k => v for (ks, v) in src for k in (ks isa Vector ? ks : [ks]))
