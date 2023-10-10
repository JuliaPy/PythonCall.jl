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
    @testset "type" begin
        @test pyis(pytype(pyjl(Foo(1))), PythonCall.pyjlanytype)
        @test pyis(pytype(pyjl(nothing)), PythonCall.pyjlanytype)
        @test pyis(pytype(pyjl(missing)), PythonCall.pyjlanytype)
    end
    @testset "bool" begin
        @test pytruth(pyjl(Foo(0)))
        @test pytruth(pyjl(Foo(1)))
        @test pytruth(pyjl(nothing))
        @test pytruth(pyjl(missing))
    end
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

@testitem "array" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(fill(nothing))), PythonCall.pyjlarraytype)
        @test pyis(pytype(pyjl([1 2; 3 4])), PythonCall.pyjlarraytype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl(fill(nothing, 0, 1)))
        @test !pytruth(pyjl(fill(nothing, 1, 0)))
        @test pytruth(pyjl(fill(nothing)))
        @test pytruth(pyjl(fill(nothing, 1, 2)))
        @test pytruth(pyjl(fill(nothing, 1, 2, 3)))
    end
end

@testitem "base" begin

end

@testitem "callback" begin

end

@testitem "dict" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(Dict())), PythonCall.pyjldicttype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl(Dict()))
        @test pytruth(pyjl(Dict("one"=>1, "two"=>2)))
    end
end

@testitem "io" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(devnull)), PythonCall.pyjlbinaryiotype)
        @test pyis(pytype(pybinaryio(devnull)), PythonCall.pyjlbinaryiotype)
        @test pyis(pytype(pytextio(devnull)), PythonCall.pyjltextiotype)
    end
    @testset "bool" begin
        @test pytruth(pybinaryio(devnull))
        @test pytruth(pytextio(devnull))
    end
end

@testitem "iter" begin
    x1 = [1,2,3,4,5]
    x2 = pyjl(x1)
    x3 = pylist(x2)
    x4 = pyconvert(Vector{Int}, x3)
    @test x1 == x4
end

@testitem "module" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(PythonCall)), PythonCall.pyjlmoduletype)
    end
    @testset "bool" begin
        @test pytruth(pyjl(PythonCall))
    end
    @testset "seval" begin
        m = Py(Main)
        @test pyconvert(Any, m.seval("1 + 1")) === 2 # Basic behavior
        @test pyconvert(Any, m.seval("1 + 1\n ")) === 2 # Trailing whitespace
    end
end

@testitem "number" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(false)), PythonCall.pyjlintegertype)
        @test pyis(pytype(pyjl(0)), PythonCall.pyjlintegertype)
        @test pyis(pytype(pyjl(0//1)), PythonCall.pyjlrationaltype)
        @test pyis(pytype(pyjl(0.0)), PythonCall.pyjlrealtype)
        @test pyis(pytype(pyjl(Complex(0.0))), PythonCall.pyjlcomplextype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl(false))
        @test !pytruth(pyjl(0))
        @test !pytruth(pyjl(0//1))
        @test !pytruth(pyjl(0.0))
        @test !pytruth(pyjl(Complex(0.0)))
        @test pytruth(pyjl(true))
        @test pytruth(pyjl(3))
        @test pytruth(pyjl(5//2))
        @test pytruth(pyjl(2.3))
        @test pytruth(pyjl(Complex(1.2, 3.4)))
    end
end

@testitem "objectarray" begin

end

@testitem "raw" begin

end

@testitem "set" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(Set())), PythonCall.pyjlsettype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl(Set()))
        @test pytruth(pyjl(Set([1,2,3])))
    end
end

@testitem "type" begin
    @testset "type" begin
        @test pyis(pytype(pyjl(Int)), PythonCall.pyjltypetype)
    end
    @testset "bool" begin
        @test pytruth(pyjl(Int))
    end
end

@testitem "vector" begin
    @testset "type" begin
        @test pyis(pytype(pyjl([1, 2, 3, 4])), PythonCall.pyjlvectortype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl([]))
        @test pytruth(pyjl([1]))
        @test pytruth(pyjl([1,2]))
    end
end
