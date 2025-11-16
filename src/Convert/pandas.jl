pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function init_pandas()
    pyconvert_add_rule(
        pyconvert_rule_pandas_na,
        "pandas._libs.missing:NAType",
        Missing,
        Any,
    )
    pyconvert_add_rule(pyconvert_rule_pandas_na, "pandas._libs.missing:NAType", Nothing, Nothing)
end
