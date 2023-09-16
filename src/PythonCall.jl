module PythonCall

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(@__DIR__)

include("CPython/CPython.jl")
const C = _CPython

end
