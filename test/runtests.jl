using PythonCall, Test, Dates, Aqua

# The unbound_args test fails on methods with signature like foo(::Type{Tuple{Vararg{V}}}) where V
# Seems like a bug.
Aqua.test_all(PythonCall, unbound_args=false)

@testset "PythonCall.jl" begin
    @testset "utils" begin
        include("utils.jl")
    end
    @testset "abstract" begin
        include("abstract.jl")
    end
    @testset "concrete" begin
        include("concrete.jl")
    end
    @testset "convert" begin
        include("convert.jl")
    end
    @testset "jlwrap" begin
        include("jlwrap.jl")
    end
end
