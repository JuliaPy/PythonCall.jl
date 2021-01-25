using Python, Test, Dates

@testset "Python.jl" begin

    @testset "cpython" begin
    end

    @testset "eval" begin
        @pyg ```
        import sys, os, datetime
        class Foo:
            def __init__(self, x=None):
                self.x = x
            def __repr__(self):
                return "Foo()"
            def __str__(self):
                return "<some Foo>"
        ```
    end

    @testset "convert-to-python" begin
        for v in [nothing, missing]
            @test @pyv `$v is None`::Bool
        end
        @test @pyv `$true is True`::Bool
        @test @pyv `$false is False`::Bool
        for T in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt]
            @test @pyv `type($(zero(T))) is int`::Bool
        end
        for T in [Float16,Float32,Float64]
            @test @pyv `type($(zero(T))) is float`::Bool
        end
        for T in [Float16,Float32,Float64]
            @test @pyv `type($(zero(Complex{T}))) is complex`::Bool
        end
        @test @pyv `type($("hello")) is str`::Bool
        @test @pyv `type($(SubString("hello", 1, 2))) is str`::Bool
        @test @pyv `type($('a')) is str`::Bool
        @test @pyv `type($(())) is tuple`::Bool
        @test @pyv `type($((1,2,"foo",nothing))) is tuple`::Bool
        @test @pyv `type($(Date(2000,1,1))) is datetime.date`::Bool
        @test @pyv `type($(Time(12,0,0))) is datetime.time`::Bool
        @test @pyv `type($(DateTime(2000,1,1,12,0,0))) is datetime.datetime`::Bool
        for T in [Second, Millisecond, Microsecond, Nanosecond]
            @test @pyv `type($(T(0))) is datetime.timedelta`::Bool
        end
        for v in Any[zero(BigFloat), (x=1, y=nothing), IOBuffer(), [], [1,2,3], Dict(), Set(), [1 2; 3 4]]
            @test pyisjl(PyObject(v))
        end
    end

    @testset "builtins" begin
        Foo = @pyv `Foo`
        foo = @pyv `Foo()`
        @test pyis(nothing, nothing)
        @test pyis(foo, foo)
        @test !pyis(foo, @pyv `Foo()`)
        @test pyhasattr(foo, :x)
        @test !pyhasattr(foo, :y)
        @test pyhasattr(foo, "x")
        @test !pyhasattr(foo, "y")
        @test pysetattr(foo, :x, 1) === foo
        @test pysetattr(foo, "y", 2) === foo
        @test pyhasattr(foo, "x")
        @test pyhasattr(foo, :y)
        @test @pyv `$(pygetattr(foo, "x")) is $foo.x`::Bool
        @test @pyv `$(pygetattr(foo, :y)) is $foo.y`::Bool
        @test @pyv `"x" in $(pydir(foo))`::Bool
        @test @pyv `"y" in $(pydir(foo))`::Bool
        @test @pyv `"z" not in $(pydir(foo))`::Bool
        @test @pyv `type($(pycall(Foo))) is Foo`::Bool
        @test @pyv `$(pyrepr(foo)) == "Foo()"`::Bool
        @test @pyv `$(pystr(foo)) == "<some Foo>"`::Bool
        @test @pyv `$(pybytes(Int8[])) == b""`::Bool
        @test @pyv `$(pybytes(pybytes(Int8[]))) == b""`::Bool
        @test pylen(@pyv `[]`) == 0
        list = @pyv `[1,2,3]`
        @test pylen(list) == 3
        @test pycontains(list, 3)
        @test !pycontains(list, 4)
        @test 3 ∈ list
        @test 4 ∉ list
        @test @pyv `$(pygetitem(list, 2)) == 3`::Bool
        @test pysetitem(list, 2, 99) === list
        @test @pyv `$(pygetitem(list, 2)) == 99`::Bool
        dict = @pyv `dict()`
        @test "x" ∉ dict
        @test pysetitem(dict, "x", "foo") === dict
        @test "x" ∈ dict
        @test pylen(dict) == 1
        @test @pyv `$(pygetitem(dict, "x")) == "foo"`::Bool
        @test pydelitem(dict, "x") === dict
        @test "x" ∉ dict
        @test pylen(dict) == 0
        @test @pyv `$(pynone()) is None`::Bool
        @test @pyv `$(pybool(false)) is False`::Bool
        @test @pyv `$(pybool(true)) is True`::Bool
        @test @pyv `$(pybool(0)) is False`::Bool
        @test @pyv `$(pybool(1)) is True`::Bool
        @test @pyv `$(pybool("")) is False`::Bool
        @test @pyv `$(pybool("foo")) is True`::Bool
        @test @pyv `$(pybool(())) is False`::Bool
        @test @pyv `$(pybool((0,))) is True`::Bool
        @test @pyv `$(pybool((1,2,3))) is True`::Bool
    end

    @testset "julia" begin
        for value in Any[nothing, false, true, 0, 1, typemin(Int), 1.0, (), (1,false,()), [], Int[1,2,3], stdin, stdout, IOBuffer()]
            r = pyjlraw(value)
            @test r isa PyObject
            @test pyisjl(r)
            @test pyjlgetvalue(r) === value
            v = pyjl(value)
            @test v isa PyObject
            @test pyisjl(v)
            @test pyjlgetvalue(v) === value
        end

        @testset "juliaany" begin
            for value in Any[nothing, missing, (), identity, push!]
                @test @pyv `type($(pyjl(value))).__name__ == "AnyValue"`::Bool
            end
        end

        @testset "juliaio" begin
            for value in Any[stdin, stdout, IOBuffer()]
                @test @pyv `type($(pyjl(value))).__name__ == "BufferedIOValue"`::Bool
                @test @pyv `type($(pyrawio(value))).__name__ == "RawIOValue"`::Bool
                @test @pyv `type($(pybufferedio(value))).__name__ == "BufferedIOValue"`::Bool
                @test @pyv `type($(pytextio(value))).__name__ == "TextIOValue"`::Bool
            end
        end
    end

end
