@testitem "PyArray" begin
end

@testitem "PyDict" begin
    x = pydict(["foo"=>12])
    y = PyDict(x)
    z = PyDict{String,Int}(x)
    @testset "construct" begin
        @test y isa PyDict{Py,Py}
        @test PyDict{String}(x) isa PyDict{String,Py}
        @test z isa PyDict{String,Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "length" begin
        @test length(y) == 1
        @test length(z) == 1
    end
    @testset "copy" begin
        t = copy(z)
        @test t isa PyDict{String,Int}
        @test !pyis(t, z)
        t["bar"] = 34
        @test z == Dict("foo"=>12)
        @test t == Dict("foo"=>12, "bar"=>34)
    end
    @testset "iterate" begin
        @test collect(z) == ["foo" => 12]
    end
    @testset "getindex" begin
        @test z["foo"] === 12
        @test_throws KeyError z["bar"]
    end
    @testset "setindex!" begin
        t = copy(z)
        @test setindex!(t, 34, "bar") === t
        @test t == Dict("foo"=>12, "bar"=>34)
        @test setindex!(t, 56, "foo") === t
        @test t == Dict("foo"=>56, "bar"=>34)
        @test_throws Exception setindex!(t, 0, nothing)
        @test_throws Exception setindex!(t, nothing, "foo")
        @test t == Dict("foo"=>56, "bar"=>34)
    end
    @testset "delete!" begin
        t = copy(z)
        @test delete!(t, "bar") === t
        @test t == Dict("foo"=>12)
        @test delete!(t, 0) === t
        @test t == Dict("foo"=>12)
        @test delete!(t, "foo") === t
        @test isempty(t)
        @test delete!(t, "foo") === t
        @test isempty(t)
    end
    @testset "empty!" begin
        t = copy(z)
        @test !isempty(t)
        @test empty!(t) === t
        @test isempty(t)
    end
    @testset "haskey" begin
        @test haskey(z, "foo")
        @test !haskey(z, "bar")
        @test !haskey(z, nothing)
        @test !haskey(z, 99)
    end
    @testset "get" begin
        t = copy(z)
        @test get(t, "foo", nothing) === 12
        @test get(t, "bar", nothing) === nothing
        @test get(t, nothing, missing) === missing
        @test get(t, 0, 1) === 1
        @test get(Vector, t, "foo") === 12
        @test get(Vector, t, "bar") == []
        @test t == Dict("foo"=>12)
    end
    @testset "get!" begin
        t = copy(z)
        @test get!(t, "foo", 0) === 12
        @test t == Dict("foo"=>12)
        @test get!(t, "bar", 0) === 0
        @test t == Dict("foo"=>12, "bar"=>0)
        @test get!(()->99, t, "foo") === 12
        @test t == Dict("foo"=>12, "bar"=>0)
        @test get!(()->99, t, "baz") === 99
        @test t == Dict("foo"=>12, "bar"=>0, "baz"=>99)
        @test_throws Exception get!(t, 0, 0)
        @test_throws Exception get!(t, "", "")
        @test_throws Exception get!(()->99, t, 0)
        @test_throws Exception get!(Vector, t, "")
        @test t == Dict("foo"=>12, "bar"=>0, "baz"=>99)
    end
    @testset "construct empty" begin
        @test PyDict() isa PyDict{Py,Py}
        @test PyDict{String}() isa PyDict{String,Py}
        @test PyDict{String,Int}() isa PyDict{String,Int}
        @test isempty(PyDict{String,Int}())
    end
end

@testitem "PyIO" begin
end

@testitem "PyIterable" begin
    x = pylist([1, 2, 3])
    y = PyIterable(x)
    z = PyIterable{Int}(x)
    @testset "construct" begin
        @test y isa PyIterable{Py}
        @test z isa PyIterable{Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "iterate" begin
        @test Base.IteratorSize(typeof(y)) === Base.SizeUnknown()
        @test Base.IteratorSize(typeof(z)) === Base.SizeUnknown()
        @test eltype(y) == Py
        @test eltype(z) == Int
        @test collect(z) == [1, 2, 3]
    end
end

@testitem "PyList" begin
    x = pylist([1, 2, 3])
    y = PyList(x)
    z = PyList{Int}(x)
    @testset "construct" begin
        @test y isa PyList{Py}
        @test z isa PyList{Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "length" begin
        @test length(y) == 3
        @test length(z) == 3
    end
    @testset "size" begin
        @test size(y) == (3,)
        @test size(z) == (3,)
    end
    @testset "getindex" begin
        @test y[1] isa Py
        @test pyeq(Bool, y[1], 1)
        @test pyeq(Bool, y[2], 2)
        @test pyeq(Bool, y[3], 3)
        @test_throws BoundsError y[-1]
        @test_throws BoundsError y[0]
        @test_throws BoundsError y[4]
        @test z[1] === 1
        @test z[2] === 2
        @test z[3] === 3
        @test_throws BoundsError z[-1]
        @test_throws BoundsError z[0]
        @test_throws BoundsError z[4]
    end
    @testset "copy" begin
        t = copy(z)
        @test t isa PyList{Int}
        push!(t, 99)
        @test z == [1, 2, 3]
        @test t == [1, 2, 3, 99]
    end
    @testset "setindex!" begin
        t = copy(z)
        @test setindex!(t, 11, 1) === t
        @test setindex!(t, 22.0, 2) === t
        @test setindex!(t, 66//2, 3) == t
        @test t == [11, 22, 33]
        @test_throws BoundsError t[-1] = 0
        @test_throws BoundsError t[0] = 0
        @test_throws BoundsError t[4] = 0
        @test t == [11, 22, 33]
        @test_throws Exception t[1] = nothing
        @test_throws Exception t[2] = missing
        @test_throws Exception t[3] = 4.5
        @test t == [11, 22, 33]
    end
    @testset "insert!" begin
        t = copy(z)
        @test insert!(t, 2, 11) === t
        @test t == [1, 11, 2, 3]
        @test insert!(t, 5, 33) === t
        @test t == [1, 11, 2, 3, 33]
        @test_throws BoundsError insert!(t, -1, 0)
        @test_throws BoundsError insert!(t, 0, 0)
        @test_throws BoundsError insert!(t, 7, 0)
        @test t == [1, 11, 2, 3, 33]
        @test_throws Exception insert!(t, nothing, 2)
        @test t == [1, 11, 2, 3, 33]
    end
    @testset "push!" begin
        t = copy(z)
        @test push!(t, 4) === t
        @test t == [1, 2, 3, 4]
        @test push!(t, 5, 6) === t
        @test t == [1, 2, 3, 4, 5, 6]
        @test_throws Exception push!(t, missing)
        @test t == [1, 2, 3, 4, 5, 6]
    end
    @testset "pushfirst!" begin
        t = copy(z)
        @test pushfirst!(t, -1) === t
        @test t == [-1, 1, 2, 3]
        @test pushfirst!(t, -3, -2) === t
        @test t == [-3, -2, -1, 1, 2, 3]
        @test_throws Exception pushfirst!(t, 4.5)
        @test t == [-3, -2, -1, 1, 2, 3]
    end
    @testset "append!" begin
        t = copy(z)
        @test append!(t, [4, 5, 6]) === t
        @test t == [1, 2, 3, 4, 5, 6]
        @test_throws Exception append!(t, [nothing, missing])
        @test t == [1, 2, 3, 4, 5, 6]
    end
    @testset "pop!" begin
        t = copy(z)
        @test pop!(t) == 3
        @test pop!(t) == 2
        @test pop!(t) == 1
        @test isempty(t)
        @test_throws BoundsError pop!(t)
    end
    @testset "popfirst!" begin
        t = copy(z)
        @test popfirst!(t) == 1
        @test popfirst!(t) == 2
        @test popfirst!(t) == 3
        @test isempty(t)
        @test_throws BoundsError popfirst!(t)
    end
    @testset "popat!" begin
        t = copy(z)
        @test_throws BoundsError popat!(t, 0)
        @test_throws BoundsError popat!(t, 4)
        @test t == [1, 2, 3]
        @test popat!(t, 2) == 2
        @test popat!(t, 2) == 3
        @test popat!(t, 1) == 1
        @test isempty(t)
        @test_throws BoundsError popat!(t, 1)
        @test_throws BoundsError popat!(t, 0)
        @test_throws BoundsError popat!(t, 1)
        @test_throws BoundsError popat!(t, 5)
    end
    @testset "reverse!" begin
        t = copy(z)
        @test reverse!(t) === t
        @test t == [3, 2, 1]
    end
    @testset "empty!" begin
        t = copy(z)
        @test !isempty(t)
        @test empty!(t) === t
        @test isempty(t)
    end
    @testset "construct empty" begin
        t = PyList{Int}()
        @test t isa PyList{Int}
        @test isempty(t)
        @test pyisinstance(t, pybuiltins.list)
    end
end

@testitem "PyPandasDataFrame" begin
end

@testitem "PySet" begin
    @testset "copy" begin
        x = PySet{Int}([1,2,3])
        y = copy(x)
        @test y isa PySet{Int}
        push!(y, 99)
        @test x == Set([1, 2, 3])
        @test y == Set([1, 2, 3, 99])
    end
end

@testitem "PyTable" begin
end
