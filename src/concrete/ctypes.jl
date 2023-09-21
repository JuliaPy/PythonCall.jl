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
    for (t,T) in CTYPES_SIMPLE_TYPES
        isptr = endswith(t, "_p")
        isreal = !isptr
        isnumber = isreal
        isfloat = t in ("float", "double")
        isint = isreal && !isfloat
        isuint = isint && (startswith(t, "u") || t == "size_t")

        name = "ctypes:c_$t"
        rule = pyconvert_rule_ctypessimplevalue{T, false}()
        saferule = pyconvert_rule_ctypessimplevalue{T, true}()

        t == "char_p" && pyconvert_add_rule(saferule, name, Cstring)
        t == "wchar_p" && pyconvert_add_rule(saferule, name, Cwstring)
        pyconvert_add_rule(saferule, name, T)
        isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(UInt) ? saferule : rule, name, UInt)
        isuint && pyconvert_add_rule(sizeof(T) < sizeof(Int) ? saferule : rule, name, Int)
        isint && !isuint && pyconvert_add_rule(sizeof(T) ≤ sizeof(Int) ? saferule : rule, name, Int)
        isint && pyconvert_add_rule(rule, name, Integer)
        isfloat && pyconvert_add_rule(saferule, name, Float64)
        isreal && pyconvert_add_rule(rule, name, Real)
        isnumber && pyconvert_add_rule(rule, name, Number)
        isptr && pyconvert_add_rule(saferule, name, Ptr)
    end
end
