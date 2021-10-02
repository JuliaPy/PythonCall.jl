using PythonCall, Test, Dates, Aqua

Aqua.test_all(PythonCall)

@testset "PythonCall.jl" begin
    @testset "abstract" begin
        include("abstract.jl")
    end
    @testset "concrete" begin
        include("concrete.jl")
    end
end
