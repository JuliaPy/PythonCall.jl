pyconvert_rule_pandas_na(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_pandas_na(::Type{Missing}, x::Py) = pyconvert_return(missing)

