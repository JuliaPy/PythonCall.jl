pyconvert_rule_bytes(::Type{Vector{UInt8}}, x::Py) = pyconvert_return(copy(pybytes_asvector(x)))
pyconvert_rule_bytes(::Type{Base.CodeUnits{UInt8,String}}, x::Py) = pyconvert_return(codeunits(pybytes_asUTF8string(x)))
