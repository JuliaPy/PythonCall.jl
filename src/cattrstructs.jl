@kwdef struct CPyMethodDefStruct
    name :: Cstring = C_NULL
    meth :: Ptr{Cvoid} = C_NULL
    flags :: Cint = 0
    doc :: Cstring = C_NULL
end

const CPy_METH_VARARGS = 0x0001 # args are a tuple of arguments
const CPy_METH_KEYWORDS = 0x0002  # two arguments: the varargs and the kwargs
const CPy_METH_NOARGS = 0x0004  # no arguments (NULL argument pointer)
const CPy_METH_O = 0x0008       # single argument (not wrapped in tuple)
const CPy_METH_CLASS = 0x0010 # for class methods
const CPy_METH_STATIC = 0x0020 # for static methods

@kwdef struct CPyGetSetDefStruct
    name :: Cstring = C_NULL
    get :: Ptr{Cvoid} = C_NULL
    set :: Ptr{Cvoid} = C_NULL
    doc :: Cstring = C_NULL
    closure :: Ptr{Cvoid} = C_NULL
end

@kwdef struct CPyMemberDefStruct
    name :: Cstring = C_NULL
    typ :: Cint = C_NULL
    offset :: CPy_ssize_t = 0
    flags :: Cint = 0
    doc :: Cstring = C_NULL
end

const CPy_T_SHORT        =0
const CPy_T_INT          =1
const CPy_T_LONG         =2
const CPy_T_FLOAT        =3
const CPy_T_DOUBLE       =4
const CPy_T_STRING       =5
const CPy_T_OBJECT       =6
const CPy_T_CHAR         =7
const CPy_T_BYTE         =8
const CPy_T_UBYTE        =9
const CPy_T_USHORT       =10
const CPy_T_UINT         =11
const CPy_T_ULONG        =12
const CPy_T_STRING_INPLACE       =13
const CPy_T_BOOL         =14
const CPy_T_OBJECT_EX    =16
const CPy_T_LONGLONG     =17 # added in Python 2.5
const CPy_T_ULONGLONG    =18 # added in Python 2.5
const CPy_T_PYSSIZET     =19 # added in Python 2.6
const CPy_T_NONE         =20 # added in Python 3.0

const CPy_READONLY = 1
const CPy_READ_RESTRICTED = 2
const CPy_WRITE_RESTRICTED = 4
const CPy_RESTRICTED = (CPy_READ_RESTRICTED | CPy_WRITE_RESTRICTED)
