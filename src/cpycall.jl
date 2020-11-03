cpycall_arg(x) = x

cpycall_argtype(::Type{T}) where {T} = T
cpycall_argtype(::Type{Base.RefValue{T}}) where {T} = Ptr{T}
cpycall_argtype(::Type{T}) where {T<:AbstractString} = Cstring

cpycall_raw(f, ::Type{T}, args...) where {T}  =
    cpycall_raw_inner(f, T, map(cpycall_arg, args)...)

@generated cpycall_raw_inner(::Val{f}, ::Type{T}, args...) where {f, T} =
    :(ccall(
        ($(QuoteNode(f)), PYLIB),
        $T,
        ($(map(cpycall_argtype, args)...),),
        $([:(args[$i]) for i in 1:length(args)]...)
    ))

cpycall_voidx(f, args...) = cpycall_raw(f, Cvoid, args...)
cpycall_intx(f, args...) = cpycall_raw(f, Cint, args...)
cpycall_boolx(f, args...) = cpycall_intx(f, args...) != 0
cpycall_stringx(f, args...) = unsafe_string(cpycall_raw(f, Cstring, args...))
cpycall_wstringx(f, args...) = unsafe_string(cpycall_raw(f, Cwstring, args...))
