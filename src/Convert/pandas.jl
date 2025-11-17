pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function pandas_rule_specs()
    return PyConvertRuleSpec[
        (func = pyconvert_rule_pandas_na, tname = "pandas._libs.missing:NAType", type = Missing, scope = Any),
        (
            func = pyconvert_rule_pandas_na,
            tname = "pandas._libs.missing:NAType",
            type = Nothing,
            scope = Nothing,
        ),
    ]
end
