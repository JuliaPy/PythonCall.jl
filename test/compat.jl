@testitem "pywith" begin
    @testset "no error" begin
        tdir = pyimport("tempfile").TemporaryDirectory()
        tname = pyconvert(String, tdir.name)
        @test isdir(tname)
        pywith(tdir) do name
            @test pyconvert(String, name) == tname
        end
        @test !isdir(tname)
    end
    @testset "error" begin
        tdir = pyimport("tempfile").TemporaryDirectory()
        tname = pyconvert(String, tdir.name)
        @test isdir(tname)
        @test_throws PyException pywith(name -> name.invalid_attr, tdir)
        @test !isdir(tname)
    end
end
