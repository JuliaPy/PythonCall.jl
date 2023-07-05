pyisnone(x) = pyis(x, pybuiltins.None)

pyconvert_rule_none(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_none(::Type{Missing}, x::Py) = pyconvert_return(missing)
