function cpycheckbuffer(o::CPyPtr)
    p = UnsafePtr{CPyTypeObject}(cpytype(o)).as_buffer[]
    p != C_NULL && p.get[] != C_NULL
end

function cpygetbuffer(o::CPyPtr, b::Ptr{CPy_buffer}, flags::Integer)
    p = UnsafePtr{CPyTypeObject}(cpytype(o)).as_buffer[]
    if p == C_NULL || p.get[] == C_NULL
        pyerrset(pytypeerror, "a bytes-like object is required, not '$(cpytypename(cpytype(o)))'")
        return Cint(-1)
    end
    ccall(p.get[!], Cint, (CPyPtr, Ptr{CPy_buffer}, Cint), o, b, flags)
end

function cpyreleasebuffer(b::Ptr{CPy_buffer})
    o = UnsafePtr(b).obj[!]
    o == C_NULL && return
    p = UnsafePtr{CPyTypeObject}(cpytype(o)).as_buffer[]
    if (p != C_NULL && p.release[] != C_NULL)
        ccall(p.release[!], Cvoid, (CPyPtr, Ptr{CPy_buffer}), o, b)
    end
    UnsafePtr(b).obj[] = C_NULL
    cpydecref(o)
    return
end

"""
    PyBuffer(o)

A reference to the underlying buffer of `o`, if it satisfies the buffer protocol.

Has the following properties:
- `buf`: Pointer to the data.
- `obj`: The exporting object (usually `o`).
- `len`: The length of the buffer in bytes.
- `readonly`: True if the buffer is immutable.
- `itemsize`: The size of each element.
- `format`: The struct-syntax format of the element type.
- `ndim`: The number of dimensions.
- `shape`: The length of the buffer in each dimension.
- `strides`: The strides (in bytes) of the buffer in each dimension.
- `suboffsets`: For indirect arrays. See the buffer protocol documentation.
- `isccontiguous`: True if the buffer is C-contiguous (e.g. numpy arrays).
- `isfcontiguous`: True if the buffer is Fortran-contiguous (e.g. Julia arrays).
- `eltype`: The element type.
"""
mutable struct PyBuffer
    info :: Array{CPy_buffer, 0}
    function PyBuffer(o::AbstractPyObject, flags::Integer=CPyBUF_FULL_RO)
        info = fill(CPy_buffer())
        cpygetbuffer(pyptr(o), pointer(info), flags) == 0 || pythrow()
        b = new(info)
        finalizer(b) do b
            cpyreleasebuffer(pointer(b.info))
        end
        b
    end
end
export PyBuffer

Base.getproperty(b::PyBuffer, k::Symbol) =
    if k == :buf
        b.info[].buf
    elseif k == :obj
        pynewobject(b.info[].obj, true)
    elseif k == :len
        b.info[].len
    elseif k == :readonly
        !iszero(b.info[].readonly)
    elseif k == :itemsize
        b.info[].itemsize
    elseif k == :format
        p = b.info[].format
        p == C_NULL ? "B" : unsafe_string(p)
    elseif k == :ndim
        b.info[].ndim
    elseif k == :shape
        p = b.info[].shape
        p == C_NULL ? (fld(b.len, b.itemsize),) : ntuple(i->unsafe_load(p, i), b.ndim)
    elseif k == :strides
        p = b.info[].strides
        p == C_NULL ? size_to_cstrides(b.itemsize, b.shape...) : ntuple(i->unsafe_load(p, i), b.ndim)
    elseif k == :suboffsets
        p = b.info[].suboffsets
        p == C_NULL ? ntuple(i->-1, b.ndim) : ntuple(i->unsafe_load(p, i), b.ndim)
    elseif k == :isccontiguous
        b.strides == size_to_cstrides(b.itemsize, b.shape...)
    elseif k == :isfcontiguous
        b.strides == size_to_fstrides(b.itemsize, b.shape...)
    elseif k == :eltype
        pybufferformat_to_type(b.format)
    else
        getfield(b, k)
    end
