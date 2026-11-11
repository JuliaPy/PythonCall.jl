# Run in the benchmark environment with an optional output label.
# Val keeps the requested type visible to the benchmark caller's inference.
using PythonCall, BenchmarkTools, InteractiveUtils, Profile
const ScalarConvert = PythonCall.Convert
scalar_convert(::Val{T}, x) where T = pyconvert(T, x, nothing)
scalar_sink(::Val{T}, x) where T = (pyconvert(T, x, nothing); nothing)
scalar_try(::Val{T}, x) where T = pytryconvert(T, x)
function scalar_plain(::Val{T}, x) where T
    @pyregion ScalarConvert.pytryconvert_plain(PyConvertContext(), T, x)
end
function scalar_first(::Val{T}, x) where T
    @pyregion begin
        ctx = PyConvertContext()
        selection = ScalarConvert.pyconvert_selection(ctx, T, PythonCall.C.Py_Type(x))
        ScalarConvert.pyconvert_first(ctx, T, x, selection)
    end
end

label = isempty(ARGS) ? "current" : first(ARGS)
fixtures = [("int", Int, pyint(42)), ("float", Float64, pyfloat(1.5)),
            ("bigint", BigInt, pyint(big(2)^200)),
            ("none", Nothing, pybuiltins.None)]
open(joinpath(@__DIR__, "convert-scalars-$label.tsv"), "w") do io
    println(io, "case\tpath\tmin_ns\tmedian_ns\tbytes\tallocs")
    for (name, T, x) in fixtures
        tag = Val(T)
        scalar_convert(tag, x)
        dynamic = @benchmark pyconvert($T, $x, nothing) seconds=0.5 samples=2000
        m, d = minimum(dynamic), median(dynamic)
        println(io, "$name\tdynamic\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
        for (path, f) in (("typed", scalar_convert), ("discard", scalar_sink),
                          ("try", scalar_try), ("plain", scalar_plain),
                          ("first", scalar_first))
            trial = @benchmark $f($tag, $x) seconds=0.5 samples=2000
            m, d = minimum(trial), median(trial)
            println(io, "$name\t$path\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
        end
        flush(io)
    end
end
open(joinpath(@__DIR__, "convert-scalars-$label-inference.txt"), "w") do io
    for (name, T, x) in fixtures, f in (scalar_convert, scalar_plain, scalar_first)
        println(io, "\n$name: $f")
        inferred = IOBuffer()
        code_warntype(inferred, f, Tuple{Val{T},Py}; debuginfo=:none)
        println(io, rstrip(String(take!(inferred))))
    end
end
open(joinpath(@__DIR__, "convert-scalars-$label-allocations.txt"), "w") do io
    for (name, T, x) in fixtures
        tag = Val(T)
        scalar_convert(tag,x)
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 scalar_convert(tag,x)
        println(io, "\n$name")
        for a in Profile.Allocs.fetch().allocs
            println(io, "$(a.type): $(a.size) bytes")
            for frame in a.stacktrace
                file = string(frame.file)
                if occursin("PythonCall",file) || occursin("gmp",file) || occursin("boot.jl",file)
                    println(io, "  ", frame)
                end
            end
        end
    end
end

# Interleave the important public calls so one unusually fast/slow short trial
# does not determine the comparison of None with Int. Include an Int outside
# Julia's small boxed-integer cache to expose otherwise hidden boxing.
open(joinpath(@__DIR__, "convert-scalars-$label-repeat.tsv"), "w") do io
    println(io,"round\tcase\tmin_ns\tmedian_ns\tbytes\tallocs")
    repeated=[fixtures; [("int_large",Int,pyint(10000))]]
    for round=1:3, (name,T,x) in circshift(repeated,round-1)
        tag=Val(T)
        trial=@benchmark scalar_convert($tag,$x) seconds=0.5 samples=5000
        m,d=minimum(trial),median(trial)
        println(io,"$round\t$name\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
    end
end
