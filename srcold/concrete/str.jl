pyconvert_rule_str(::Type{String}, x::Py) = pyconvert_return(pystr_asstring(x))
pyconvert_rule_str(::Type{Symbol}, x::Py) = pyconvert_return(Symbol(pystr_asstring(x)))
pyconvert_rule_str(::Type{Char}, x::Py) = begin
    s = pystr_asstring(x)
    if length(s) == 1
        pyconvert_return(first(s))
    else
        pyconvert_unconverted()
    end
end
