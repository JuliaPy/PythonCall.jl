const pyinttype = PyLazyObject(() -> pybuiltins.int)
export pyinttype

pyint(args...; opts...) = pyinttype(args...; opts...)
pyint(x::Integer) =
    if typemin(Clonglong) ≤ x ≤ typemax(Clonglong)
        cpycall_obj(Val(:PyLong_FromLongLong), convert(Clonglong, x))
    else
        # TODO: it's probably faster to do this in base 16
        cpycall_obj(Val(:PyLong_FromString), string(convert(BigInt, x)), C_NULL, Cint(10))
    end
pyint(x::Unsigned) =
    if x ≤ typemax(Culonglong)
        cpycall_obj(Val(:PyLong_FromUnsignedLongLong), convert(Culonglong, x))
    else
        pyint(BigInt(x))
    end
export pyint

pyisint(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_LONG_SUBCLASS)
export pyisint

function pyint_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    if BigInt <: T
        # if it fits in a longlong, use that
        rl = cpycall_raw(Val(:PyLong_AsLongLong), Clonglong, o)
        if rl != -1 || !pyerroccurred()
            return BigInt(rl)
        elseif !pyerroccurred(pyoverflowerror)
            pythrow()
        else
            pyerrclear()
            return parse(BigInt, pystr(String, o))
        end
    elseif (S = _typeintersect(T, Integer)) != Union{}
        if S <: Unsigned
            # if it fits in a ulonglong, use that
            rl = cpycall_raw(Val(:PyLong_AsUnsignedLongLong), Culonglong, o)
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
            rl = cpycall_raw(Val(:PyLong_AsLongLong), Clonglong, o)
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
