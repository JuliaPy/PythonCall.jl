struct pyconvert_rule_ctypessimplevalue{R,S} <: Function end

function (::pyconvert_rule_ctypessimplevalue{R,SAFE})(::Type{T}, x::Py) where {R,SAFE,T}
    Base.GC.@preserve x begin
        ptr = C.PySimpleObject_GetValue(Ptr{R}, x)
        ans = unsafe_load(ptr)
        if SAFE
            pyconvert_return(convert(T, ans))
        else
            pyconvert_tryconvert(T, ans)
        end
    end
end

const CTYPES_SIMPLE_TYPES = [
    ("char", Cchar),
    ("wchar", Cwchar_t),
    ("byte", Cchar),
    ("ubyte", Cuchar),
    ("short", Cshort),
    ("ushort", Cushort),
    ("int", Cint),
    ("uint", Cuint),
    ("long", Clong),
    ("ulong", Culong),
    ("longlong", Clonglong),
    ("ulonglong", Culonglong),
    ("size_t", Csize_t),
    ("ssize_t", Cssize_t),
    ("float", Cfloat),
    ("double", Cdouble),
    ("char_p", Ptr{Cchar}),
    ("wchar_p", Ptr{Cwchar_t}),
    ("void_p", Ptr{Cvoid}),
]

function init_ctypes()
    for (t, T) in CTYPES_SIMPLE_TYPES
        isptr = endswith(t, "_p")
        isnumber = !isptr
        isfloat = t in ("float", "double")
        isint = isnumber && !isfloat
        isuint = isint && (startswith(t, "u") || t == "size_t")

        name = "ctypes:c_$t"
        rule = pyconvert_rule_ctypessimplevalue{T,false}()
        saferule = pyconvert_rule_ctypessimplevalue{T,true}()

        if isnumber
            # Rules added later are tried first. Prefer the source's exact Julia type,
            # followed by increasingly general lossless representations, before allowing
            # conversion to floating-point and other number types.
            pyconvert_add_rule(name, Number, Number, rule)
            pyconvert_add_rule(name, Real, Number, rule)
            pyconvert_add_rule(name, AbstractFloat, Number, rule)
            if isint
                pyconvert_add_rule(name, Integer, Number, rule)
                pyconvert_add_rule(name, isuint ? Unsigned : Signed, Number, rule)
                if isuint
                    pyconvert_add_rule(
                        name,
                        UInt,
                        Number,
                        sizeof(T) ≤ sizeof(UInt) ? saferule : rule,
                    )
                else
                    pyconvert_add_rule(
                        name,
                        Int,
                        Number,
                        sizeof(T) ≤ sizeof(Int) ? saferule : rule,
                    )
                end
            elseif isfloat
                pyconvert_add_rule(name, Float64, Number, saferule)
            end
            pyconvert_add_rule(name, T, Number, saferule)
        elseif isptr
            pyconvert_add_rule(name, Ptr, Ptr, saferule)
            pyconvert_add_rule(name, T, Ptr, saferule)
        end
    end
end
