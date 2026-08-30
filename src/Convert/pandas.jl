pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function init_pandas()
    pyconvert_add_rule_high_priority(
        "pandas.api.typing:NAType",
        Missing,
        Any,
        pyconvert_rule_pandas_na,
        0,
    )
    pyconvert_add_rule("pandas.api.typing:NAType", Nothing, Nothing, pyconvert_rule_pandas_na)
end
