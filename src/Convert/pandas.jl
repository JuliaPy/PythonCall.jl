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

function register_pandas_rules!()
    for rule in pandas_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
end
