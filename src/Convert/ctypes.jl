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

function ctypes_rule_specs()
    specs = PyConvertRuleSpec[]
    for (t, T) in CTYPES_SIMPLE_TYPES
        isptr = endswith(t, "_p")
        isreal = !isptr
        isnumber = isreal
        isfloat = t in ("float", "double")
        isint = isreal && !isfloat
        isuint = isint && (startswith(t, "u") || t == "size_t")

        name = "ctypes:c_$t"
        rule = pyconvert_rule_ctypessimplevalue{T,false}()
        saferule = pyconvert_rule_ctypessimplevalue{T,true}()

        t == "char_p" && push!(specs, (func = saferule, tname = name, type = Cstring, scope = Cstring))
        t == "wchar_p" && push!(specs, (func = saferule, tname = name, type = Cwstring, scope = Cwstring))
        push!(specs, (func = saferule, tname = name, type = T, scope = T))
        isuint && push!(
            specs,
            (func = sizeof(T) ≤ sizeof(UInt) ? saferule : rule, tname = name, type = UInt, scope = UInt),
        )
        isuint && push!(
            specs,
            (func = sizeof(T) < sizeof(Int) ? saferule : rule, tname = name, type = Int, scope = Int),
        )
        isint && !isuint && push!(
            specs,
            (func = sizeof(T) ≤ sizeof(Int) ? saferule : rule, tname = name, type = Int, scope = Int),
        )
        isint && push!(specs, (func = rule, tname = name, type = Integer, scope = Integer))
        isfloat && push!(specs, (func = saferule, tname = name, type = Float64, scope = Float64))
        isreal && push!(specs, (func = rule, tname = name, type = Real, scope = Real))
        isnumber && push!(specs, (func = rule, tname = name, type = Number, scope = Number))
        isptr && push!(specs, (func = saferule, tname = name, type = Ptr, scope = Ptr))
    end
    return specs
end

