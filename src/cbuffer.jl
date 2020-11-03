Base.@kwdef struct CPyBufferProcs
    get :: Ptr{Cvoid} # (o, Ptr{CPy_buffer}, Cint) -> Cint
    release :: Ptr{Cvoid} # (o, Ptr{CPy_buffer}) -> Cvoid
end

Base.@kwdef struct CPy_buffer
    buf :: Ptr{Cvoid} = C_NULL
    obj :: Ptr{Cvoid} = C_NULL
    len :: CPy_ssize_t = 0
    itemsize :: CPy_ssize_t = 0
    readonly :: Cint = 0
    ndim :: Cint = 0
    format :: Cstring = C_NULL
    shape :: Ptr{CPy_ssize_t} = C_NULL
    strides :: Ptr{CPy_ssize_t} = C_NULL
    suboffsets :: Ptr{CPy_ssize_t} = C_NULL
    internal :: Ptr{Cvoid} = C_NULL
end

const CPyBUF_MAX_NDIM = 64

# Flags for getting buffers
const CPyBUF_SIMPLE = 0x0
const CPyBUF_WRITABLE = 0x0001
const CPyBUF_WRITEABLE = CPyBUF_WRITABLE
const CPyBUF_FORMAT = 0x0004
const CPyBUF_ND = 0x0008
const CPyBUF_STRIDES = (0x0010 | CPyBUF_ND)
const CPyBUF_C_CONTIGUOUS = (0x0020 | CPyBUF_STRIDES)
const CPyBUF_F_CONTIGUOUS = (0x0040 | CPyBUF_STRIDES)
const CPyBUF_ANY_CONTIGUOUS = (0x0080 | CPyBUF_STRIDES)
const CPyBUF_INDIRECT = (0x0100 | CPyBUF_STRIDES)

const CPyBUF_CONTIG = (CPyBUF_ND | CPyBUF_WRITABLE)
const CPyBUF_CONTIG_RO = (CPyBUF_ND)

const CPyBUF_STRIDED = (CPyBUF_STRIDES | CPyBUF_WRITABLE)
const CPyBUF_STRIDED_RO = (CPyBUF_STRIDES)

const CPyBUF_RECORDS = (CPyBUF_STRIDES | CPyBUF_WRITABLE | CPyBUF_FORMAT)
const CPyBUF_RECORDS_RO = (CPyBUF_STRIDES | CPyBUF_FORMAT)

const CPyBUF_FULL = (CPyBUF_INDIRECT | CPyBUF_WRITABLE | CPyBUF_FORMAT)
const CPyBUF_FULL_RO = (CPyBUF_INDIRECT | CPyBUF_FORMAT)

const CPyBUF_READ = 0x100
const CPyBUF_WRITE = 0x200
