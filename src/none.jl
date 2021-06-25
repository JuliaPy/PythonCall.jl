pyisnone(x) = pyis(x, pybuiltins.None)

pyconvert_rule_none(::Type{Nothing}, x::Py) = pyconvert_return(nothing)
pyconvert_rule_none(::Type{Missing}, x::Py) = pyconvert_return(missing)

pyconvert_rule_fast(::Type{Nothing}, x::Py) =
    if pyisnone(x)
        pyconvert_return(nothing)
    else
        pyconvert_unconverted()
    end

pyconvert_rule_fast(::Type{Missing}, x::Py) =
    if pyisnone(x)
        pyconvert_return(missing)
    else
        pyconvert_unconverted()
    end
