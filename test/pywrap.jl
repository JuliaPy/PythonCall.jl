@testset "PyArray" begin
end

@testset "PyDict" begin
    @testset "copy" begin
        x = PyDict{String,Int}()
        x["foo"] = 12
        y = copy(x)
        @test y isa PyDict{String,Int}
        y["bar"] = 34
        @test x == Dict("foo"=>12)
        @test y == Dict("foo"=>12, "bar"=>34)
    end
end

@testset "PyIO" begin
end

@testset "PyIterable" begin
end

@testset "PyList" begin
    @testset "copy" begin
        x = PyList{Int}([1, 2, 3])
        y = copy(x)
        @test y isa PyList{Int}
        push!(y, 99)
        @test x == [1, 2, 3]
        @test y == [1, 2, 3, 99]
    end
end

@testset "PyPandasDataFrame" begin
end

@testset "PySet" begin
    @testset "copy" begin
        x = PySet{Int}([1,2,3])
        y = copy(x)
        @test y isa PySet{Int}
        push!(y, 99)
        @test x == Set([1, 2, 3])
        @test y == Set([1, 2, 3, 99])
    end
end

@testset "PyTable" begin
end
