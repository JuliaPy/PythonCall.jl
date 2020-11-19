module Python

using Dates, UnsafePointers, Libdl, Conda, Tables, TableTraits, IteratorInterfaceExtensions, Markdown
using Base: @kwdef

# dependencies
include(joinpath(@__DIR__, "..", "deps", "deps.jl"))

# things not directly dependent on PyObject or libpython
include("utils.jl")

# C API
include("cpython.jl")

const C = CPython
const CPyPtr = C.PyPtr
struct CPyObjRef
    ptr :: CPyPtr
end

# initialize
include("init.jl")

# core
include("object.jl")
include("error.jl")
include("import.jl")

# abstract interfaces
include("number.jl")
include("sequence.jl")

# fundamental objects
include("type.jl")
include("none.jl")

# numeric objects
include("bool.jl")
include("int.jl")
include("float.jl")
include("complex.jl")

# sequence objects
include("str.jl")
include("bytes.jl")
include("bytearray.jl")
include("tuple.jl")
include("list.jl")

# mapping objects
include("dict.jl")
include("set.jl")

# other objects
include("slice.jl")
include("range.jl")

# standard library
include("builtins.jl")
include("stdlib.jl")
include("io.jl")
include("fraction.jl")
include("datetime.jl")
include("collections.jl")

# other packages
include("pandas.jl")
include("numpy.jl")

# other Julia wrappers around Python values
include("PyIterable.jl")
include("PyList.jl")
include("PyDict.jl")
include("PyObjectArray.jl")
include("PyBuffer.jl")
include("PyArray.jl")

# other functionality
include("newtype.jl")
include("julia.jl")
include("base.jl")
include("pywith.jl")
include("macros.jl")

# otherwise forward declarations are required
include("convert.jl")

end # module
