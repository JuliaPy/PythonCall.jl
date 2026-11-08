pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function init_pandas()
    # pandas.NA is public across supported versions, even where NAType is not.
    pyconvert_add_rule(
        "pandas:NA.__class__",
        Missing,
        pyconvert_rule_pandas_na,
        ; scope=Any,
    )
    pyconvert_add_rule("pandas:NA.__class__", Nothing, pyconvert_rule_pandas_na)
end
