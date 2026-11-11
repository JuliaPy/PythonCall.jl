# Run with julia --project=benchmark benchmark/convert-redesign.jl baseline|redesign
using PythonCall, BenchmarkTools
label = isempty(ARGS) ? "redesign" : ARGS[1]
const CV = PythonCall.Convert
cases = [
    ("int", Int, pyint(42)), ("integer", Integer, pyint(42)),
    ("bigint", BigInt, pyint(big(2)^200)), ("float", Float64, pyfloat(1.5)),
    ("bool", Bool, pybool(true)), ("string", String, pystr("hello")),
    ("none", Nothing, pybuiltins.None), ("failure", Int, pystr("oops")),
    ("list10", Vector, pylist(1:10)), ("list10000", Vector, pylist(1:10000)),
    ("mixed", Vector, pylist([1, 2.5, "three", nothing])),
    ("nested", Any, pylist([pylist(1:5), pylist(6:10)])),
    ("dict", Dict, pydict("a"=>1, "b"=>2.5)),
    ("wrapper", PyList, pylist(1:10)),
    ("iterator", Any, pyiter(pylist(1:10))),
]
open(joinpath(@__DIR__, "convert-$label.tsv"), "w") do io
    println(io, "case\tns\tbytes\tallocs")
    for (name, T, x) in cases
        trial = @benchmark pyconvert($T, $x, nothing) seconds=0.3 samples=1000
        m = minimum(trial)
        println(io, "$name\t$(m.time)\t$(m.memory)\t$(m.allocs)")
    end
end
