cpycall_argtype(::Type{T}) where {T<:AbstractPyObject} = CPyPtr
Base.unsafe_convert(::Type{CPyPtr}, o::AbstractPyObject) = pyptr(o)

function cpycall_ref(f, args...)
    r = cpycall_raw(f, CPyPtr, args...)
    r == C_NULL ? pythrow() : r
end

cpycall_obj(f, args...) = pynewobject(cpycall_ref(f, args...))

function cpycall_num(f, ::Type{T}, args...) where {T}
    r = cpycall_raw(f, T, args...)
    r == (zero(T) - one(T)) ? pythrow() : r
end

cpycall_int(f, args...) = cpycall_num(f, Cint, args...)

cpycall_bool(f, args...) = cpycall_int(f, args...) != 0

cpycall_void(f, args...) = (cpycall_int(f, args...); nothing)

function cpycall_num_ambig(f, ::Type{T}, args...) where {T}
    r = cpycall_raw(f, T, args...)
    r == (zero(T) - one(T)) && pyerrcheck()
    return r
end
