module Tables

using ...PythonCall

using Tables: Tables as T
using Requires: @require

import ...PythonCall: pytable


asvector(x::AbstractVector) = x
asvector(x) = collect(x)

"""
    pytable(src, format=:pandas; ...)

Construct a Python table from the Tables.jl-compatible table `src`.

The `format` controls the type of the resulting table, and is one of:
- `:pandas`: A `pandas.DataFrame`. Keyword arguments are passed to the `DataFrame` constructor.
- `:columns`: A `dict` mapping column names to columns.
- `:rows`: A `list` of rows, which are `namedtuple`s.
- `:rowdicts`: A `list` of rows, which are `dict`s.
"""
function pytable(src, format = :pandas; opts...)
    format = Symbol(format)
    if format == :pandas
        _pytable_pandas(src; opts...)
    elseif format == :columns
        _pytable_columns(src)
    elseif format == :rows
        _pytable_rows(src)
    elseif format == :rowdicts
        _pytable_rowdicts(src)
    else
        error("invalid format")
    end
end

function _pytable_columns(src, cols = T.columns(src))
    pydict(pystr(String(n)) => asvector(T.getcolumn(cols, n)) for n in T.columnnames(cols))
end

function _pytable_rows(src, rows = T.rows(src))
    names = T.columnnames(rows)
    t = pyimport("collections" => "namedtuple")(
        "Row",
        pylist(pystr(string(n)) for n in names),
    )
    pylist(t(map(n -> T.getcolumn(row, n), names)...) for row in rows)
end

function _pytable_rowdicts(src, rows = T.rows(src))
    names = T.columnnames(rows)
    pynames = [pystr(string(n)) for n in names]
    pylist(
        pydict(p => T.getcolumn(row, n) for (n, p) in zip(names, pynames)) for row in rows
    )
end

aspandasvector(x) = asvector(x)

function _pytable_pandas(src, cols = T.columns(src); opts...)
    pyimport("pandas").DataFrame(
        pydict(
            pystr(string(n)) => aspandasvector(T.getcolumn(cols, n)) for
            n in T.columnnames(cols)
        );
        opts...,
    )
end

function __init__()
    @require CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597" @eval begin
        aspandasvector(x::CategoricalArrays.CategoricalArray) = begin
            codes = map(x -> x === missing ? -1 : Int(CategoricalArrays.levelcode(x)) - 1, x)
            cats = CategoricalArrays.levels(x)
            ordered = x.pool.ordered
            pyimport("pandas").Categorical.from_codes(codes, cats, ordered = ordered)
        end
    end
end

end
