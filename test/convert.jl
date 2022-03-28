@testset "iterable -> Tuple" begin
    t1 = pyconvert(Tuple, (1, 2))
    @test t1 === (1, 2)
    t2 = pyconvert(Tuple{Vararg{Int}}, (3, 4, 5))
    @test t2 === (3, 4, 5)
    t3 = pyconvert(Tuple{Int,Int}, (6, 7))
    @test t3 === (6, 7)
    # generic case (>16 fields)
    t4 = pyconvert(Tuple{ntuple(i->Int,20)...,Varag{Int}}, ntuple(i->i, 30))
    @test t4 === ntuple(i->i, 30)
end

@testset "iterable -> Vector" begin
    x1 = pyconvert(Vector, pylist([1, 2]))
    @test x1 isa Vector{Int}
    @test x1 == [1, 2]
    x2 = pyconvert(Vector, pylist([1, 2, nothing, 3]))
    @test x2 isa Vector{Union{Int,Nothing}}
    @test x2 == [1, 2, nothing, 3]
    x3 = pyconvert(Vector{Float64}, pylist([4, 5, 6]))
    @test x3 isa Vector{Float64}
    @test x3 == [4.0, 5.0, 6.0]
end

@testset "iterable -> Set" begin
    x1 = pyconvert(Set, pyset([1, 2]))
    @test x1 isa Set{Int}
    @test x1 == Set([1, 2])
    x2 = pyconvert(Set, pyset([1, 2, nothing, 3]))
    @test x2 isa Set{Union{Int,Nothing}}
    @test x2 == Set([1, 2, nothing, 3])
    x3 = pyconvert(Set{Float64}, pyset([4, 5, 6]))
    @test x3 isa Set{Float64}
    @test x3 == Set([4.0, 5.0, 6.0])
end

@testset "iterable -> Pair" begin
    @test_throws Exception pyconvert(Pair, ())
    @test_throws Exception pyconvert(Pair, (1,))
    @test_throws Exception pyconvert(Pair, (1, 2, 3))
    x1 = pyconvert(Pair, (2, 3))
    @test x1 === (2 => 3)
    x2 = pyconvert(Pair{String,Missing}, ("foo", nothing))
    @test x2 === ("foo" => missing)
end

@testset "mapping -> Dict" begin
    x1 = pyconvert(Dict, pydict(["a"=>1, "b"=>2]))
    @test x1 isa Dict{String, Int}
    @test x1 == Dict("a"=>1, "b"=>2)
    x2 = pyconvert(Dict{Char,Float32}, pydict(["c"=>3, "d"=>4]))
    @test x2 isa Dict{Char,Float32}
    @test x2 == Dict('c'=>3.0, 'd'=>4.0)
end
