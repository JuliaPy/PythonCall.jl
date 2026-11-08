# Isolate result representation from Python dispatch and iteration.
# julia --project=benchmark benchmark/convert-results.jl
using BenchmarkTools, PythonCall, InteractiveUtils
label = isempty(ARGS) ? "" : "-" * only(ARGS)
struct UnconvertedBaseline end
@noinline function old_result(::Type{T}, x, success) where T
    (success ? (x::T) : UnconvertedBaseline())::Union{T,UnconvertedBaseline}
end
@noinline function new_result(::Type{T}, x, success) where T
    (success ? Some(x::T) : nothing)::some_result_returntype(T)
end
# This candidate intentionally wraps every success, independently of the public API.
function some_result_bound(T)
    T === Any && return Union{Nothing,Some}
    T isa Union && return Union{some_result_bound(T.a),some_result_bound(T.b)}
    isconcretetype(T) ? Union{Nothing,Some{T}} : Union{Nothing,Some{<:T}}
end
@generated some_result_returntype(::Type{T}) where T = QuoteNode(some_result_bound(T))
@noinline function normalized_result(::Type{T}, x, success) where T
    (success ? PythonCall.Convert.pyconvert_return(x::T) : nothing)::PythonCall.Convert.pyconvert_returntype(T)
end
@noinline escaping(x) = Ref{Any}(x)
function consume(f,::Type{T},values) where T
    result = 0
    for x in values
        r = f(T,x,true)
        result += r isa Some ? something(r) : r
    end
    result
end
cases = [(Int64,1), (Float64,1.5), (String,"hello"), (Vector{Int},[1,2]),
         (Integer,1), (AbstractArray,[1,2]), (Union{Int64,Nothing},1), (Any,1)]
open(joinpath(@__DIR__,"convert-results$label.tsv"),"w") do io
    println(io,"type\trepresentation\tsuccess\tescaping\tns\tbytes\tallocs")
    for (T,x) in cases, (name,f) in (("raw",old_result),("some",new_result),("normalized",normalized_result)), success in (true,false), escape in (true,false)
        b = escape ? (@benchmark escaping($f($T,$x,$success)) seconds=0.1 samples=500) :
                     (@benchmark $f($T,$x,$success) seconds=0.1 samples=500)
        m=minimum(b)
        println(io,"$T\t$name\t$success\t$escape\t$(m.time)\t$(m.memory)\t$(m.allocs)")
    end
    for (name,f) in (("raw",old_result),("some",new_result),("normalized",normalized_result))
        values=collect(1:100)
        m=minimum(@benchmark consume($f,Int,$values) seconds=0.1 samples=500)
        println(io,"loop100\t$name\ttrue\tfalse\t$(m.time)\t$(m.memory)\t$(m.allocs)")
    end
end
open(joinpath(@__DIR__,"convert-inference$label.txt"),"w") do io
    for T in (Int,Integer,Union{Int,Nothing},Any)
        println(io,"\npytryconvert($T, ::Py)")
        inferred=IOBuffer()
        code_warntype(inferred,pytryconvert,Tuple{Type{T},Py};debuginfo=:none)
        println(io,rstrip(String(take!(inferred))))
    end
end
