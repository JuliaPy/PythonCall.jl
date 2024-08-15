pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function init_pandas()
    pyconvert_add_rule(
        "pandas._libs.missing:NAType",
        Missing,
        pyconvert_rule_pandas_na,
        PYCONVERT_PRIORITY_CANONICAL,
    )
    pyconvert_add_rule("pandas._libs.missing:NAType", Nothing, pyconvert_rule_pandas_na)
end
