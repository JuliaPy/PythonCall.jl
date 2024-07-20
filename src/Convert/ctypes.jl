struct pyconvert_rule_ctypessimplevalue{R,S} <: Function end

function (::pyconvert_rule_ctypessimplevalue{R,SAFE})(::Type{T}, x::Py) where {R,SAFE,T}
    ptr = Base.GC.@preserve x C.PySimpleObject_GetValue(Ptr{R}, getptr(x))
    ans = unsafe_load(ptr)
    if SAFE
        pyconvert_return(convert(T, ans))
    else
        pyconvert_tryconvert(T, ans)
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
        isreal = !isptr
        isnumber = isreal
        isfloat = t in ("float", "double")
        isint = isreal && !isfloat
        isuint = isint && (startswith(t, "u") || t == "size_t")

        name = "ctypes:c_$t"
        rule = pyconvert_rule_ctypessimplevalue{T,false}()
        saferule = pyconvert_rule_ctypessimplevalue{T,true}()

        t == "char_p" && pyconvert_add_rule(name, Cstring, saferule)
        t == "wchar_p" && pyconvert_add_rule(name, Cwstring, saferule)
        pyconvert_add_rule(name, T, saferule)
        isuint && pyconvert_add_rule(name, UInt, sizeof(T) ≤ sizeof(UInt) ? saferule : rule)
        isuint && pyconvert_add_rule(name, Int, sizeof(T) < sizeof(Int) ? saferule : rule)
        isint &&
            !isuint &&
            pyconvert_add_rule(name, Int, sizeof(T) ≤ sizeof(Int) ? saferule : rule)
        isint && pyconvert_add_rule(name, Integer, rule)
        isfloat && pyconvert_add_rule(name, Float64, saferule)
        isreal && pyconvert_add_rule(name, Real, rule)
        isnumber && pyconvert_add_rule(name, Number, rule)
        isptr && pyconvert_add_rule(name, Ptr, saferule)
    end
end
