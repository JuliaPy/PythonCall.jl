@testset "iter" begin
    x1 = [1,2,3,4,5]
    x2 = pyjl(x1)
    x3 = pylist(x2)
    x4 = pyconvert(Vector{Int}, x3)
    @test x1 == x4
end

@testset "dtypes" begin
    if !pymoduleexists("numpy")
        PythonCall.C.CondaPkg.add("numpy")
    end
    np = pyimport("numpy");
    y = range(-5,5,length=11)
    arr = np.asarray(y)
    @test pyconvert(Int, arr.size) == 11
    @test pyconvert(String, arr.dtype.name) == "float64"
    @test all(iszero.(pyconvert(Any, arr) .- y))
    arr32 = np.asarray(y, dtype=np.float32)
    @test pyconvert(Int, arr32.size) == 11
    @test pyconvert(String, arr32.dtype.name) == "float32"
    @test all(iszero.(pyconvert(Any, arr32) .- y))
end
