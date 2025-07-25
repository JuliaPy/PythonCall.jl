
ispy(x::PyPandasDataFrame) = true
Py(x::PyPandasDataFrame) = x.py

pyconvert_rule_pandasdataframe(::Type{PyPandasDataFrame}, x::Py) =
    pyconvert_return(PyPandasDataFrame(x))

### Show

### Tables

Tables.istable(::Type{PyPandasDataFrame}) = true

Tables.columnaccess(::Type{PyPandasDataFrame}) = true

Tables.columns(df::PyPandasDataFrame) = _columns(df, df.columnnames, df.columntypes)

function _columns(df, columnnames, columntypes)
    # collect columns
    colnames = Symbol[]
    pycolumns = Py[]
    if df.indexname !== nothing
        push!(colnames, df.indexname)
        push!(pycolumns, df.py.index.values)
    end
    for pycolname in df.py.columns
        colname = columnnames(pycolname)::Symbol
        pycolumn = df.py[pycolname].values
        push!(colnames, colname)
        push!(pycolumns, pycolumn)
    end
    # ensure column names are unique by appending a _N suffix
    colnamecount = Dict{Symbol,Int}()
    for (i, colname) in pairs(colnames)
        n = get(colnamecount, colname, 0) + 1
        colnamecount[colname] = n
        if n > 1
            colnames[i] = Symbol(colname, :_, n)
        end
    end
    # convert columns to vectors
    columns = AbstractVector[]
    coltypes = Type[]
    for (colname, pycolumn) in zip(colnames, pycolumns)
        coltype = columntypes(colname)::Union{Nothing,Type}
        if coltype !== nothing
            column = pyconvert(AbstractVector{coltype}, pycolumn)
        else
            column = pyconvert(AbstractVector, pycolumn)
            if eltype(column) == Py
                column = pyconvert(AbstractVector{Any}, pycolumn)
            end
            if !isconcretetype(eltype(column))
                column = [(x === nothing) ? missing : x for x in column]
                if eltype(column) != Float64 && Float64 <: eltype(column)
                    column = [x === NaN ? missing : x for x in column]
                end
            end
        end
        push!(columns, column)
        push!(coltypes, eltype(column))
    end
    # output a table
    # TODO: realising columns to vectors could be done lazily with a different table type
    schema = Tables.Schema(colnames, coltypes)
    coldict = Tables.OrderedDict(k => v for (k, v) in zip(colnames, columns))
    table = Tables.DictColumnTable(schema, coldict)
    Tables.columns(table)
end
