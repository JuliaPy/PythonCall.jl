using PythonCall, Test, Dates, Aqua

Aqua.test_all(PythonCall)

@testset "PythonCall.jl" begin
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
