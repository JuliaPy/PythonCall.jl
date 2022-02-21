@testset "iterable -> Tuple" begin
    t1 = pyconvert(Tuple, (1, 2))
    @test t1 === (1, 2)
    t2 = pyconvert(Tuple{Vararg{Int}}, (3, 4, 5))
    @test t2 === (3, 4, 5)
    t3 = pyconvert(Tuple{Int,Int}, (6, 7))
    @test t3 === (6, 7)
end
