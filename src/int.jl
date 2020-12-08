const pyinttype = pylazyobject(() -> pybuiltins.int)
export pyinttype

pyint(args...; opts...) = pyinttype(args...; opts...)
pyint(x::Integer) =
    if typemin(Clonglong) ≤ x ≤ typemax(Clonglong)
        check(C.PyLong_FromLongLong(x))
    else
        # TODO: it's probably faster to do this in base 16
        check(C.PyLong_FromString(string(convert(BigInt, x)), C_NULL, 10))
    end
pyint(x::Unsigned) =
    if x ≤ typemax(Culonglong)
        check(C.PyLong_FromUnsignedLongLong(x))
    else
        pyint(BigInt(x))
    end
export pyint

pyisint(o::PyObject) = pytypecheckfast(o, C.Py_TPFLAGS_LONG_SUBCLASS)
export pyisint

function pyint_tryconvert(::Type{T}, o::PyObject) where {T}
    if BigInt <: T
        # if it fits in a longlong, use that
        rl = C.PyLong_AsLongLong(o)
        if rl != -1 || !pyerroccurred()
            # NOTE: In this case we return an Int if possible, since a lot of Julia functions take an Int but not a BigInt
            # This is the only major exception to the rule that the Python type determines the Julia type.
            return (Int <: T && typemin(Int) ≤ r1 ≤ typemax(Int)) ? Int(r1) : BigInt(rl)
        elseif !pyerroccurred(pyoverflowerror)
            pythrow()
        else
            pyerrclear()
            return parse(BigInt, pystr(String, o))
        end
    elseif (S = _typeintersect(T, Integer)) != Union{}
        if S <: Unsigned
            # if it fits in a ulonglong, use that
            rl = C.PyLong_AsUnsignedLongLong(o)
            if rl != zero(Culonglong)-one(Culonglong) || !pyerroccurred()
                return tryconvert(S, rl)
            elseif !pyerroccurred(pyoverflowerror)
                pythrow()
            elseif S in (UInt8, UInt16, UInt32, UInt64, UInt128) && sizeof(S) ≤ sizeof(Culonglong)
                pyerrclear()
                return PyConvertFail()
            end
        else
            # if it fits in a longlong, use that
            rl = C.PyLong_AsLongLong(o)
            if rl != -1 || !pyerroccurred()
                return tryconvert(S, rl)
            elseif !pyerroccurred(pyoverflowerror)
                pythrow()
            elseif S in (Int8, Int16, Int32, Int64, Int128) && sizeof(S) ≤ sizeof(Clonglong)
                pyerrclear()
                return PyConvertFail()
            end
        end
        # last resort: print to a string
        pyerrclear()
        return tryconvert(S, parse(BigInt, pystr(String, o)))
    else
        tryconvert(T, pyint_tryconvert(BigInt, o))
    end
end
