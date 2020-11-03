const pystrtype = PyLazyObject(() -> pybuiltins.str)
export pystrtype

pystr(x::Union{String,SubString{String}}) = cpycall_obj(Val(:PyUnicode_DecodeUTF8), x, CPy_ssize_t(ncodeunits(x)), C_NULL)
pystr(x::AbstractString) = pystr(convert(String, x))
pystr(x::AbstractChar) = pystr(String(x))
pystr(x::Symbol) = pystr(String(x))
pystr(args...; opts...) = pystrtype(args...; opts...)
export pystr

pystr_asutf8string(o::AbstractPyObject) = cpycall_obj(Val(:PyUnicode_AsUTF8String), o)

pystr_asjuliastring(o::AbstractPyObject) = pybytes_asjuliastring(pystr_asutf8string(o))

pyisstr(o::AbstractPyObject) = pytypecheckfast(o, CPy_TPFLAGS_UNICODE_SUBCLASS)
export pyisstr

function pystr_tryconvert(::Type{T}, o::AbstractPyObject) where {T}
    x = pystr_asjuliastring(o)
    if (S = _typeintersect(T, String)) != Union{}
        convert(S, x)
    elseif (S = _typeintersect(T, AbstractString)) != Union{}
        convert(S, x)
    elseif (S = _typeintersect(T, Symbol)) != Union{}
        convert(S, Symbol(x))
    elseif (S = _typeintersect(T, Char)) != Union{}
        length(x)==1 ? convert(S, x[1]) : PyConvertFail()
    elseif (S = _typeintersect(T, AbstractChar)) != Union{}
        length(x)==1 ? convert(S, x[1]) : PyConvertFail()
    else
        tryconvert(T, o)
    end
end
