"""
    pointer_from_obj(x)

Returns `(p, c)` where `Base.pointer_from_objref(p) === x`.

The pointer remains valid provided the object `c` is not garbage collected.
"""
function pointer_from_obj(o::T) where {T}
    if T.mutable
        c = o
        p = Base.pointer_from_objref(o)
    else
        c = Ref{Any}(o)
        p = unsafe_load(Ptr{Ptr{Cvoid}}(Base.pointer_from_objref(c)))
    end
    p, c
end

islittleendian() =
    Base.ENDIAN_BOM == 0x04030201 ? true : Base.ENDIAN_BOM == 0x01020304 ? false : error()

isflagset(flags, mask) = (flags & mask) == mask
