@testitem "iter" begin
    x1 = [1,2,3,4,5]
    x2 = pyjl(x1)
    x3 = pylist(x2)
    x4 = pyconvert(Vector{Int}, x3)
    @test x1 == x4
end
