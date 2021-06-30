pybool(x::Bool=false) = Py(x ? pybuiltins.True : pybuiltins.False)
pybool(x::Number) = pybool(!iszero(x))
pybool(x) = pybuiltins.bool(x)
export pybool

pyisTrue(x) = pyis(x, pybuiltins.True)
pyisFalse(x) = pyis(x, pybuiltins.False)
pyisbool(x) = pyisTrue(x) || pyisFalse(x)

pybool_asbool(x) =
    @autopy x if pyisTrue(x_)
        true
    elseif pyisFalse(x_)
        false
    else
        error("not a bool")
    end

function pyconvert_rule_bool(::Type{T}, x::Py) where {T<:Number}
    val = pybool_asbool(x)
    if T in (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128, BigInt)
        pyconvert_return(T(val))
    else
        pyconvert_tryconvert(T, val)
    end
end

pyconvert_rule_fast(::Type{Bool}, x::Py) =
    if pyisTrue(x)
        pyconvert_return(true)
    elseif pyisFalse(x)
        pyconvert_return(false)
    else
        pyconvert_unconverted()
    end
