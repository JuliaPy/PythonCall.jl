module PythonCall

greet(name) = println("Hi, $(name)!!!")

include("cpython/CPython.jl")
include("hpython/HPython.jl")
include("jpython/JPython.jl")

end
