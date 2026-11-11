pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

function init_pandas()
    # Singleton source locations resolve to their types without private imports.
    pyconvert_add_rule(
        "pandas:NA",
        Missing,
        pyconvert_rule_pandas_na,
        ; scope=Any,
    )
    pyconvert_add_rule("pandas:NA", Nothing, pyconvert_rule_pandas_na)
    pyconvert_add_rule("pandas:NaT", Missing, pyconvert_rule_pandas_na)
    pyconvert_add_rule("pandas:NaT", Nothing, pyconvert_rule_pandas_na)
end
