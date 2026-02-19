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

