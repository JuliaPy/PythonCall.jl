# Same harness on original v1 and the final working tree. No new-API dependency.
# julia --project=benchmark benchmark/convert-final.jl after-r1
using PythonCall, BenchmarkTools, SHA

final_value(::Val{T}, x) where T = pyconvert(T,x,nothing)
label = isempty(ARGS) ? "after-r1" : only(ARGS)
newapi = isdefined(PythonCall,:PyConvertContext)
fixtures = [
    ("int",Int,pyint(10000),x->x===10000),
    ("integer",Integer,pyint(42),x->x===42),
    ("bigint",BigInt,pyint(big(2)^200),x->x==big(2)^200),
    ("float",Float64,pyfloat(1.5),x->x===1.5),
    ("bool",Bool,pybool(true),x->x===true),
    ("string",String,pystr("hello"),x->x=="hello"),
    ("none",Nothing,pybuiltins.None,isnothing),
    ("failure",Int,pystr("oops"),isnothing),
    ("list10",Vector,pylist(1:10),x->x==collect(1:10)),
    ("list10000",Vector,pylist(1:10000),x->x==collect(1:10000)),
    ("list10000_concrete",Vector{Int},pylist(1:10000),x->x==collect(1:10000)),
    ("mixed_numeric",Vector,pylist([pyint(1),pyfloat(2.5)]),x->x==[1,2.5]),
    ("heterogeneous",Vector,pylist([pyint(1),pystr("two"),pybuiltins.None]),x->isequal(x,Any[1,"two",nothing])),
    ("nested_concrete",Vector{Vector{Int}},pylist([pylist(1:5),pylist(6:10)]),x->x==[collect(1:5),collect(6:10)]),
    # Any intentionally changes from wrapping to native recursive conversion.
    ("nested_any",Any,pylist([pylist(1:5),pylist(6:10)]),x->newapi ? x isa Vector{Vector{Int}} : x isa PyList),
    ("dict",Dict,pydict("a"=>1,"b"=>2.5),x->x==Dict("a"=>1,"b"=>2.5)),
    ("dict_concrete",Dict{String,Float64},pydict("a"=>1,"b"=>2.5),x->x==Dict("a"=>1.0,"b"=>2.5)),
    ("empty_list",Vector,pylist(),x->x isa Vector && isempty(x)),
    ("empty_dict",Dict,pydict(),x->x isa Dict && isempty(x)),
    ("wrapper",PyList,pylist(1:10),x->x isa PyList && length(x)==10),
    ("iterator",Any,pyiter(pylist(1:10)),x->x isa PyIterable),
]
open(joinpath(@__DIR__,"convert-final-$label.tsv"),"w") do io
    println(io,"case\tmode\tmin_ns\tmedian_ns\tbytes\tallocs")
    for (i,(name,T,x,valid)) in enumerate(fixtures)
        tag=Val(T)
        @assert valid(final_value(tag,x)) name
        modes=i<=8 ? ("typed","runtime_target") : ("typed",)
        for mode in modes
            trial = if mode=="typed"
                @benchmark final_value($tag,$x) seconds=1.0 samples=5000
            else
                @benchmark pyconvert($T,$x,nothing) seconds=1.0 samples=5000
            end
            m,d=minimum(trial),median(trial)
            println(io,"$name\t$mode\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
        end
        flush(io)
    end
end
open(joinpath(@__DIR__,"convert-final-$label-environment.txt"),"w") do io
    println(io,"Julia: ",VERSION)
    println(io,"Python: ",pyconvert(String,pyimport("sys").version))
    println(io,"Python executable: ",pyconvert(String,pyimport("sys").executable))
    println(io,"BenchmarkTools: ",pkgversion(BenchmarkTools))
    println(io,"PythonCall source: ",pathof(PythonCall))
    println(io,"New API: ",newapi)
    for file in ("Convert/pyconvert.jl","Convert/rules.jl","Wrap/PyArray.jl")
        println(io,file," SHA256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(PythonCall)),file)))))
    end
    for (name,T,x,_) in fixtures
        println(io,name," result type: ",typeof(final_value(Val(T),x)))
    end
end
