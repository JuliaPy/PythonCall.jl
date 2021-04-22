module PythonCall

using Dates,
    UnsafePointers,
    Libdl,
    Conda,
    Tables,
    TableTraits,
    IteratorInterfaceExtensions,
    Markdown,
    Requires,
    Compat,
    LinearAlgebra
using Base: @kwdef

# things not directly dependent on PyObject or libpython
include("utils.jl")

# function and struct definitions needed by CPython
include("forwarddefs.jl")

# C API
include("cpython/CPython.jl")

const C = CPython
const CPyPtr = C.PyPtr

import .CPython: @pydsl, @pydsl_nojlerror, @pydsl_interpret, @pydsl_expand
export @pydsl, @pydsl_nojlerror, @pydsl_interpret, @pydsl_expand

include("gil.jl")
include("eval.jl")
include("PyRef.jl")
include("PyCode.jl")
include("PyInternedString.jl")
include("PyLazyObject.jl")
include("builtins.jl")
include("PyException.jl")
include("PyObject.jl")
include("PyDict.jl")
include("PyList.jl")
include("PySet.jl")
include("PyIterable.jl")
include("PyIO.jl")
include("PyBuffer.jl")
include("PyArray.jl")
include("PyObjectArray.jl")
include("PyPandasDataFrame.jl")

include("julia.jl")
include("gui.jl")
include("matplotlib.jl")
include("ipython.jl")

include("init.jl")

const juliacall_pipdir = dirname(@__DIR__)
const juliacall_dir = joinpath(juliacall_pipdir, "juliacall")

end # module
