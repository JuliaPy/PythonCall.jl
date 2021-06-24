pybool(x::Bool=false) = Py(x ? pybuiltins.True : pybuiltins.False)
pybool(x::Number) = pybool(!iszero(x))
pybool(x) = pybuiltins.bool(x)
export pybool

pybool_asbool(x) = @autopy x (getptr(x_) == C.POINTERS._Py_TrueStruct ? true : getptr(x_) == C.POINTERS._Py_FalseStruct ? false : error("not a bool"))

function pyconvert_rule_bool(::Type{T}, x::Py) where {T<:Number}
    val = pybool_asbool(x)
    if T in (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, BigInt)
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end
