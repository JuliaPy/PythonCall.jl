"""
    PyPandasDataFrame(x; indexname=nothing, columntypes=())

Wraps the pandas DataFrame `x` as a Tables.jl-compatible table.

`indexname` is the name of the column to contain the index. It may be `nothing` to exclude the index.

`columntypes` is a mapping of column names to column element types, in case automatic deduction does not work.
"""
struct PyPandasDataFrame <: PyTable
    py::Py
    indexname::Union{String,Nothing}
    columntypes::Dict{String,Type}
    function PyPandasDataFrame(x; indexname=nothing, columntypes=())
        if indexname !== nothing
            indexname = convert(String, indexname)
        end
        columntypes = Dict{String,Type}(columntypes)
        new(Py(x), indexname, columntypes)
    end
end
export PyPandasDataFrame

ispy(x::PyPandasDataFrame) = true
getpy(x::PyPandasDataFrame) = x.py
pydel!(x::PyPandasDataFrame) = pydel!(x.py)

pyconvert_rule_pandasdataframe(::Type{PyPandasDataFrame}, x::Py) = pyconvert_return(PyPandasDataFrame(x))

### Dict interface

function Base.keys(df::PyPandasDataFrame)
    ans = String[]
    @py for c in df.columns
        if isinstance(c, str)
            @jl push!(ans, pyconvert(String, c))
            @del c
        else
            @jl error("name of column '$c' is not a string")
        end
    end
    if df.indexname !== nothing
        if df.indexname in ans
            error("dataframe already includes a column called '$(df.indexname)'")
        else
            pushfirst!(ans, df.indexname)
        end
    end
    return ans
end

Base.haskey(df::PyPandasDataFrame, k::String) = (df.indexname !== nothing && k == df.indexname) || @py k in df.columns
Base.haskey(df::PyPandasDataFrame, k::Symbol) = haskey(df, string(k))
Base.haskey(df::PyPandasDataFrame, k) = haskey(df, convert(String, k))

function Base.getindex(df::PyPandasDataFrame, k::String)
    # get the given column
    if df.indexname !== nothing && k == df.indexname
        c = @py df.index
    else
        c = @py df[k]
    end
    # convert to a vector
    if haskey(df.columntypes, k)
        ans = pyconvert_and_del(AbstractVector{df.columntypes[k]}, c)
    else
        ans = pyconvert_and_del(AbstractVector, c)
        # narrow the type
        ans = identity.(ans)
        # convert any Py to something more useful
        if Py <: eltype(ans)
            ans = [x isa Py ? pyconvert(Any, x) : x for x in ans]
        end
        # convert NaN to missing
        if eltype(ans) != Float64 && Float64 <: eltype(ans)
            ans = [x isa Float64 && isnan(x) ? missing : x for x in ans]
        end
    end
    return ans :: AbstractVector
end
Base.getindex(df::PyPandasDataFrame, k::Symbol) = getindex(df, string(k))
Base.getindex(df::PyPandasDataFrame, k) = getindex(df, convert(String, k))

Base.get(df::PyPandasDataFrame, k, d) = haskey(df, k) ? df[k] : d

Base.values(df::PyPandasDataFrame) = (df[k] for k in keys(df))

Base.pairs(df::PyPandasDataFrame) = (Pair{String, AbstractVector}(k, df[k]) for k in keys(df))

Base.getproperty(df::PyPandasDataFrame, k::Symbol) = hasfield(PyPandasDataFrame, k) ? getfield(df, k) : df[k]
Base.getproperty(df::PyPandasDataFrame, k::String) = getproperty(df, Symbol(k))

function Base.propertynames(df::PyPandasDataFrame, private::Bool=false)
    ans = Symbol.(keys(df))
    if private
        append!(ans, fieldnames(PyPandasDataFrame))
    else
        push!(ans, :indexname, :columntypes)
    end
    return ans
end

### Show

function Base.show(io::IO, mime::MIME"text/plain", df::PyPandasDataFrame)
    nrows = pyconvert_and_del(Int, @py df.shape[0])
    ncols = pyconvert_and_del(Int, @py df.shape[1])
    printstyled(io, nrows, '×', ncols, ' ', typeof(df), '\n', bold=true)
    pyshow(io, mime, df)
end

Base.show(io::IO, mime::MIME, df::PyPandasDataFrame) = pyshow(io, mime, df)
Base.show(io::IO, mime::MIME"text/csv", df::PyPandasDataFrame) = pyshow(io, mime, df)
Base.show(io::IO, mime::MIME"text/tab-separated-values", df::PyPandasDataFrame) = pyshow(io, mime, df)
Base.showable(mime::MIME, df::PyPandasDataFrame) = pyshowable(mime, df)

### Tables

Tables.istable(::Type{PyPandasDataFrame}) = true
Tables.columnaccess(::Type{PyPandasDataFrame}) = true
function Tables.columns(df::PyPandasDataFrame)
    ns = Tuple(Symbol.(keys(df)))
    cs = values(df)
    return NamedTuple{ns}(cs)
end
