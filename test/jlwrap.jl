@testitem "any" begin
    struct Foo
        value::Int
    end
    Base.:(+)(x::Foo) = "+ $(x.value)"
    Base.:(-)(x::Foo) = "- $(x.value)"
    Base.abs(x::Foo) = "abs $(x.value)"
    Base.:(~)(x::Foo) = "~ $(x.value)"
    Base.:(+)(x::Foo, y::Foo) = "$(x.value) + $(y.value)"
    Base.:(-)(x::Foo, y::Foo) = "$(x.value) - $(y.value)"
    Base.:(*)(x::Foo, y::Foo) = "$(x.value) * $(y.value)"
    Base.:(/)(x::Foo, y::Foo) = "$(x.value) / $(y.value)"
    Base.:(÷)(x::Foo, y::Foo) = "$(x.value) ÷ $(y.value)"
    Base.:(%)(x::Foo, y::Foo) = "$(x.value) % $(y.value)"
    Base.:(^)(x::Foo, y::Foo) = "$(x.value) ^ $(y.value)"
    Base.:(<<)(x::Foo, y::Foo) = "$(x.value) << $(y.value)"
    Base.:(>>)(x::Foo, y::Foo) = "$(x.value) >> $(y.value)"
    Base.:(&)(x::Foo, y::Foo) = "$(x.value) & $(y.value)"
    Base.:(|)(x::Foo, y::Foo) = "$(x.value) | $(y.value)"
    @testset "pos" begin
        z = pyjl(+Foo(1))
        @test pyconvert(String, z) == "+ 1"
    end
    @testset "neg" begin
        z = pyjl(-Foo(1))
        @test pyconvert(String, z) == "- 1"
    end
    @testset "abs" begin
        z = pyjl(abs(Foo(1)))
        @test pyconvert(String, z) == "abs 1"
    end
    @testset "inv" begin
        z = pyjl(~Foo(1))
        @test pyconvert(String, z) == "~ 1"
    end
    @testset "add" begin
        z = pyjl(Foo(1)) + pyjl(Foo(2))
        @test pyconvert(String, z) == "1 + 2"
    end
    @testset "radd" begin
        z = pyjlraw(Foo(1)) + pyjl(Foo(2))
        @test pyconvert(String, z) == "1 + 2"
    end
    @testset "sub" begin
        z = pyjl(Foo(1)) - pyjl(Foo(2))
        @test pyconvert(String, z) == "1 - 2"
    end
    @testset "rsub" begin
        z = pyjlraw(Foo(1)) - pyjl(Foo(2))
        @test pyconvert(String, z) == "1 - 2"
    end
    @testset "mul" begin
        z = pyjl(Foo(1)) * pyjl(Foo(2))
        @test pyconvert(String, z) == "1 * 2"
    end
    @testset "rmul" begin
        z = pyjlraw(Foo(1)) * pyjl(Foo(2))
        @test pyconvert(String, z) == "1 * 2"
    end
    @testset "truediv" begin
        z = pyjl(Foo(1)) / pyjl(Foo(2))
        @test pyconvert(String, z) == "1 / 2"
    end
    @testset "rtruediv" begin
        z = pyjlraw(Foo(1)) / pyjl(Foo(2))
        @test pyconvert(String, z) == "1 / 2"
    end
    @testset "floordiv" begin
        z = pyjl(Foo(1)) ÷ pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ÷ 2"
    end
    @testset "rfloordiv" begin
        z = pyjlraw(Foo(1)) ÷ pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ÷ 2"
    end
    @testset "mod" begin
        z = pyjl(Foo(1)) % pyjl(Foo(2))
        @test pyconvert(String, z) == "1 % 2"
    end
    @testset "rmod" begin
        z = pyjlraw(Foo(1)) % pyjl(Foo(2))
        @test pyconvert(String, z) == "1 % 2"
    end
    @testset "pow" begin
        z = pyjl(Foo(1)) ^ pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ^ 2"
    end
    @testset "rpow" begin
        z = pyjlraw(Foo(1)) ^ pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ^ 2"
    end
    @testset "lshift" begin
        z = pyjl(Foo(1)) << pyjl(Foo(2))
        @test pyconvert(String, z) == "1 << 2"
    end
    @testset "rlshift" begin
        z = pyjlraw(Foo(1)) << pyjl(Foo(2))
        @test pyconvert(String, z) == "1 << 2"
    end
    @testset "rshift" begin
        z = pyjl(Foo(1)) >> pyjl(Foo(2))
        @test pyconvert(String, z) == "1 >> 2"
    end
    @testset "rrshift" begin
        z = pyjlraw(Foo(1)) >> pyjl(Foo(2))
        @test pyconvert(String, z) == "1 >> 2"
    end
    @testset "and" begin
        z = pyjl(Foo(1)) & pyjl(Foo(2))
        @test pyconvert(String, z) == "1 & 2"
    end
    @testset "rand" begin
        z = pyjlraw(Foo(1)) & pyjl(Foo(2))
        @test pyconvert(String, z) == "1 & 2"
    end
    @testset "or" begin
        z = pyjl(Foo(1)) | pyjl(Foo(2))
        @test pyconvert(String, z) == "1 | 2"
    end
    @testset "ror" begin
        z = pyjlraw(Foo(1)) | pyjl(Foo(2))
        @test pyconvert(String, z) == "1 | 2"
    end
end

@testitem "iter" begin
    x1 = [1,2,3,4,5]
    x2 = pyjl(x1)
    x3 = pylist(x2)
    x4 = pyconvert(Vector{Int}, x3)
    @test x1 == x4
end
