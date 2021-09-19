using PythonCall, Test, Dates, Compat

@testset "PythonCall.jl" begin
    @testset "abstract" begin
        include("abstract.jl")
    end
    @testset "concrete" begin
        include("concrete.jl")
    end
end
