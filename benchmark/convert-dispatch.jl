using PythonCall, BenchmarkTools
const CV=PythonCall.Convert
label = isempty(ARGS) ? "" : "-" * only(ARGS)
x=pylist([pylist(1:5),pylist(6:10)])
scoped=pylist(1:10)
np=pyimport("numpy")
cases = [
    ("no_preference", ()->pyconvert(Any,x)),
    ("one_preference", ()->pyconvert(Any,x;prefer=("builtins:list"=>PyList,))),
    ("multiple_preferences", ()->pyconvert(Any,x;prefer=("builtins:dict"=>PyDict,"builtins:list"=>PyList))),
    ("unmatched_preferences", ()->pyconvert(Any,x;prefer=("builtins:dict"=>PyDict,))),
    ("nested_context", ()->pyconvert(PyConvertContext(;prefer=("builtins:list"=>PyList,)),Vector,x)),
    ("ndarray_native", let a=np.arange(1000); ()->pyconvert(Any,a) end),
    ("ndarray_wrapper", let a=np.arange(1000); ()->pyconvert(PyArray,a) end),
    ("bytes_explicit", let a=pybytes(pylist([1,2,3])); ()->pyconvert(Vector{UInt8},a) end),
    ("scope_filtered", ()->pyconvert(Vector{Int},scoped)),
]
open(joinpath(@__DIR__,"convert-dispatch$label.tsv"),"w") do io
    println(io,"case\tns\tbytes\tallocs")
    for (name,f) in cases
        m=minimum(@benchmark $f() seconds=1.0 samples=5000)
        println(io,"$name\t$(m.time)\t$(m.memory)\t$(m.allocs)")
    end
end
# Cache misses/invalidation are measured with untimed setup in convert-expanded.jl.
