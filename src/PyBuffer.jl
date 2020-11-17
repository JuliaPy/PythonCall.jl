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
    info :: Array{C.Py_buffer, 0}
    function PyBuffer(o::AbstractPyObject, flags::Integer=C.PyBUF_FULL_RO)
        info = fill(C.Py_buffer())
        check(C.PyObject_GetBuffer(o, pointer(info), flags))
        b = new(info)
        finalizer(b) do b
            check(C.PyBuffer_Release(pointer(b.info)))
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
