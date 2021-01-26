using Python, Test, Dates

@testset "Python.jl" begin

    @testset "cpython" begin
    end

    @testset "eval" begin
        @pyg ```
        import sys, os, datetime
        eq = lambda a, b: type(a) is type(b) and a == b
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
            @test @pyv `eq($v, None)`::Bool
        end
        @test @pyv `eq($true, True)`::Bool
        @test @pyv `eq($false, False)`::Bool
        for T in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt]
            @test @pyv `eq($(zero(T)), 0)`::Bool
        end
        for T in [Float16,Float32,Float64]
            @test @pyv `eq($(zero(T)), 0.0)`::Bool
        end
        for T in [Float16,Float32,Float64]
            @test @pyv `eq($(zero(Complex{T})), complex(0,0))`::Bool
        end
        @test @pyv `eq($("hello"), "hello")`::Bool
        @test @pyv `eq($(SubString("hello", 1, 2)), "he")`::Bool
        @test @pyv `eq($('a'), "a")`::Bool
        @test @pyv `eq($(()), ())`::Bool
        @test @pyv `eq($((1,2,"foo",nothing)), (1,2,"foo",None))`::Bool
        @test @pyv `eq($(Date(2000,1,1)), datetime.date(2000,1,1))`::Bool
        @test @pyv `eq($(Time(12,0,0)), datetime.time(12,0,0))`::Bool
        @test @pyv `eq($(DateTime(2000,1,1,12,0,0)), datetime.datetime(2000,1,1,12,0,0))`::Bool
        for T in [Second, Millisecond, Microsecond, Nanosecond]
            @test @pyv `eq($(T(0)), datetime.timedelta())`::Bool
        end
        for v in Any[zero(BigFloat), (x=1, y=nothing), IOBuffer(), [], [1,2,3], Dict(), Set(), [1 2; 3 4]]
            @test pyisjl(PyObject(v))
        end
    end

    @testset "convert-to-julia" begin
    end

    @testset "builtins" begin
        Foo = @pyv `Foo`
        foo = @pyv `Foo()`
        xstr = @pyv `"x"`
        ystr = @pyv `"y"`
        # convert
        @test pyconvert(Int, @pyv `12`) === 12
        # hasattr, getattr, setattr, delattr
        @test pyhasattr(foo, :x)
        @test !pyhasattr(foo, :y)
        @test pyhasattr(foo, "x")
        @test !pyhasattr(foo, "y")
        @test pyhasattr(foo, xstr)
        @test !pyhasattr(foo, ystr)
        @test pysetattr(foo, :x, 1) === foo
        @test pysetattr(foo, "y", 2) === foo
        @test pyhasattr(foo, "x")
        @test pyhasattr(foo, :y)
        @test pysetattr(foo, ystr, 3) === foo
        @test @pyv `$(pygetattr(foo, "x")) is $foo.x`::Bool
        @test @pyv `$(pygetattr(foo, :y)) is $foo.y`::Bool
        @test @pyv `$(pygetattr(foo, xstr)) is $foo.x`::Bool
        # dir
        @test @pyv `"x" in $(pydir(foo))`::Bool
        @test @pyv `"y" in $(pydir(foo))`::Bool
        @test @pyv `"z" not in $(pydir(foo))`::Bool
        # call
        @test @pyv `type($(pycall(Foo))) is Foo`::Bool
        # repr
        @test @pyv `eq($(pyrepr(foo)), "Foo()")`::Bool
        # str
        @test @pyv `eq($(pystr(foo)), "<some Foo>")`::Bool
        @test @pyv `eq($(pystr("foo")), "foo")`::Bool
        # bytes
        @test @pyv `eq($(pybytes(Int8[])), b"")`::Bool
        @test @pyv `eq($(pybytes(pybytes(Int8[]))), b"")`::Bool
        # len, contains, in, getitem, setitem, delitem
        @test pylen(@pyv `[]`) == 0
        list = @pyv `[1,2,3]`
        @test pylen(list) == 3
        @test pycontains(list, 3)
        @test !pycontains(list, 4)
        @test pyin(1, list)
        @test !pyin(0, list)
        @test 3 ∈ list
        @test 4 ∉ list
        @test @pyv `eq($(pygetitem(list, 2)), 3)`::Bool
        @test pysetitem(list, 2, 99) === list
        @test @pyv `eq($(pygetitem(list, 2)), 99)`::Bool
        dict = @pyv `dict()`
        @test "x" ∉ dict
        @test pysetitem(dict, "x", "foo") === dict
        @test "x" ∈ dict
        @test pylen(dict) == 1
        @test @pyv `eq($(pygetitem(dict, "x")), "foo")`::Bool
        @test pydelitem(dict, "x") === dict
        @test "x" ∉ dict
        @test pylen(dict) == 0
        # none
        @test @pyv `$(pynone()) is None`::Bool
        # bool
        @test @pyv `$(pybool(false)) is False`::Bool
        @test @pyv `$(pybool(true)) is True`::Bool
        @test @pyv `$(pybool(0)) is False`::Bool
        @test @pyv `$(pybool(1)) is True`::Bool
        @test @pyv `$(pybool("")) is False`::Bool
        @test @pyv `$(pybool("foo")) is True`::Bool
        @test @pyv `$(pybool(())) is False`::Bool
        @test @pyv `$(pybool((0,))) is True`::Bool
        @test @pyv `$(pybool((1,2,3))) is True`::Bool
        # int
        @test @pyv `eq($(pyint(true)), 1)`::Bool
        @test @pyv `eq($(pyint(false)), 0)`::Bool
        for T in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt]
            x = pyint(T(123))
            @test @pyv `eq($x, 123)`::Bool
            @test @pyv `eq($(pyint(x)), 123)`::Bool
        end
        @test @pyv `eq($(pyint()), 0)`::Bool
        # float
        for T in [Float16,Float32,Float64]
            x = pyfloat(T(12.3))
            @test @pyv `type($x) is float`::Bool
            @test @pyv `abs($x - 12.3) < 0.01`::Bool
            @test @pyv `$(pyfloat(x)) == $x`::Bool
        end
        @test @pyv `eq($(pyfloat("12.3")), 12.3)`::Bool
        # import
        sysstr = @pyv `"sys"`
        @test @pyv `$(pyimport("sys")) is sys`::Bool
        @test @pyv `$(pyimport(sysstr)) is sys`::Bool
        @test @pyv `$(pyimport("sys"=>"stdout")) is sys.stdout`::Bool
        @test @pyv `all(a is getattr(sys,b) for (a,b) in zip($(pyimport("sys"=>("stdout","stdin"))), ["stdout","stdin"]))`::Bool
        @test @pyv `all(a is b for (a,b) in zip($(pyimport("sys","os")), [sys,os]))`::Bool
        # is, truth, issubclass, isinstance
        @test pyis(nothing, nothing)
        @test pyis(foo, foo)
        @test !pyis(foo, @pyv `Foo()`)
        @test !pytruth(@pyv `[]`)
        @test pytruth(@pyv `[1,2,3]`)
        @test pyissubclass(@pyv(`bool`), @pyv(`int`))
        @test !pyissubclass(@pyv(`bool`), @pyv(`list`))
        @test pyisinstance(@pyv(`True`), @pyv(`int`))
        @test !pyisinstance(@pyv(`True`), @pyv(`list`))
        # comparison
        @test @pyv `eq($(pyhash("foo")), hash("foo"))`::Bool
        @test @pyv `$(pyeq(1,2)) is False`::Bool
        @test @pyv `$(pyne(1,2)) is True`::Bool
        @test @pyv `$(pyle(1,2)) is True`::Bool
        @test @pyv `$(pylt(1,2)) is True`::Bool
        @test @pyv `$(pyge(1,2)) is False`::Bool
        @test @pyv `$(pygt(1,2)) is False`::Bool
        @test !pyeq(Bool, 1, 2)
        @test pyne(Bool, 1, 2)
        @test pyle(Bool, 1, 2)
        @test pylt(Bool, 1, 2)
        @test !pyge(Bool, 1, 2)
        @test !pygt(Bool, 1, 2)
        # arithmetic
        @test @pyv `eq($(pyadd(1,2)), 3)`::Bool
        @test @pyv `eq($(pyiadd(1,2)), 3)`::Bool
        @test @pyv `eq($(pysub(1,2)), -1)`::Bool
        @test @pyv `eq($(pyisub(1,2)), -1)`::Bool
        @test @pyv `eq($(pymul(1,2)), 2)`::Bool
        @test @pyv `eq($(pyimul(1,2)), 2)`::Bool
        @test @pyv `eq($(pyfloordiv(1,2)), 0)`::Bool
        @test @pyv `eq($(pyifloordiv(1,2)), 0)`::Bool
        @test @pyv `eq($(pytruediv(1,2)), 0.5)`::Bool
        @test @pyv `eq($(pyitruediv(1.0,2)), 0.5)`::Bool
        @test @pyv `eq($(pyrem(1,2)), 1)`::Bool
        @test @pyv `eq($(pyirem(1,2)), 1)`::Bool
        @test @pyv `eq($(pydivmod(1,2)), (0,1))`::Bool
        @test @pyv `eq($(pylshift(1,2)), 4)`::Bool
        @test @pyv `eq($(pyilshift(1,2)), 4)`::Bool
        @test @pyv `eq($(pyrshift(1,2)), 0)`::Bool
        @test @pyv `eq($(pyirshift(1,2)), 0)`::Bool
        @test @pyv `eq($(pyand(1,2)), 0)`::Bool
        @test @pyv `eq($(pyiand(1,2)), 0)`::Bool
        @test @pyv `eq($(pyxor(1,2)), 3)`::Bool
        @test @pyv `eq($(pyixor(1,2)), 3)`::Bool
        @test @pyv `eq($(pyor(1,2)), 3)`::Bool
        @test @pyv `eq($(pyior(1,2)), 3)`::Bool
        @test @pyv `eq($(pypow(2,3)), 8)`::Bool
        @test @pyv `eq($(pypow(2,3,5)), 3)`::Bool
        @test @pyv `eq($(pyipow(2,3)), 8)`::Bool
        @test @pyv `eq($(pyipow(2,3,5)), 3)`::Bool
        @test @pyv `eq($(pyneg(2)), -2)`::Bool
        @test @pyv `eq($(pypos(2)), 2)`::Bool
        @test @pyv `eq($(pyinv(2)), ~2)`::Bool
        # iter
        @test @pyv `eq($(pyiter(list)).__next__(), 1)`::Bool
        # tuple
        list = @pyv `[1,2,3]`
        @test @pyv `eq($(pytuple(list)), (1,2,3))`::Bool
        @test @pyv `eq($(pytuple([1,2,3])), (1,2,3))`::Bool
        @test @pyv `eq($(pytuple()), ())`::Bool
        # list
        @test @pyv `eq($(pylist(list)), [1,2,3])`::Bool
        @test @pyv `eq($(pylist([1,2,3])), [1,2,3])`::Bool
        @test @pyv `eq($(pylist()), [])`::Bool
        # pycollist
        @test @pyv `eq($(pycollist(fill(1))), 1)`::Bool
        @test @pyv `eq($(pycollist([1,2])), [1,2])`::Bool
        @test @pyv `eq($(pycollist([1 2; 3 4])), [[1,3],[2,4]])`::Bool
        # pyrowlist
        @test @pyv `eq($(pyrowlist(fill(1))), 1)`::Bool
        @test @pyv `eq($(pyrowlist([1,2])), [1,2])`::Bool
        @test @pyv `eq($(pyrowlist([1 2; 3 4])), [[1,2],[3,4]])`::Bool
        # set
        @test @pyv `eq($(pyset(list)), {1,2,3})`::Bool
        @test @pyv `eq($(pyset([1,2,3])), {1,2,3})`::Bool
        @test @pyv `eq($(pyset()), set())`::Bool
        # frozenset
        @test @pyv `eq($(pyfrozenset(list)), frozenset({1,2,3}))`::Bool
        @test @pyv `eq($(pyfrozenset([1,2,3])), frozenset({1,2,3}))`::Bool
        @test @pyv `eq($(pyfrozenset()), frozenset())`::Bool
        # dict
        @test @pyv `eq($(pydict(dict)), dict())`::Bool
        @test @pyv `eq($(pydict(Dict(1=>2))), {1:2})`::Bool
        @test @pyv `eq($(pydict()), dict())`::Bool
        # slice
        @test @pyv `eq($(pyslice(1)), slice(1))`::Bool
        @test @pyv `eq($(pyslice(1,2)), slice(1,2))`::Bool
        @test @pyv `eq($(pyslice(1,2,3)), slice(1,2,3))`::Bool
        # ellipsis
        @test @pyv `$(pyellipsis()) is Ellipsis`::Bool
        # NotImplemented
        @test @pyv `$(pynotimplemented()) is NotImplemented`::Bool
        # method
        @test pymethod(identity) isa PyObject
        # type
        @test @pyv `$(pytype(3)) is int`::Bool
        @test @pyv `issubclass($(pytype("Foo", (pytype(3),), ())), int)`::Bool
    end

    @testset "PyObject" begin
        x = PyObject(1)
        y = Python.pylazyobject(()->error("nope"))
        foo = @pyv `Foo()`
        Foo = @pyv `Foo`
        list = @pyv `[1,2,3]`
        @test x isa PyObject
        @test y isa PyObject
        @test foo isa PyObject
        @test Python.pyptr(y) === Python.C.PyNULL
        Python.C.PyErr_Clear()
        @test string(x) == "1"
        @test (io=IOBuffer(); print(io, x); String(take!(io))) == "1"
        @test repr(x) == "<py 1>"
        @test (io=IOBuffer(); show(io, x); String(take!(io))) == "<py 1>"
        @test (io=IOBuffer(); show(io, MIME("text/plain"), x); String(take!(io))) == "py: 1"
        @test @pyv `$(foo.x) is $foo.x`::Bool
        @test (foo.x = 99) === 99
        @test @pyv `eq($(foo.x), 99)`::Bool
        @test @pyv `type($(Foo())) is Foo`::Bool
        @test Base.IteratorSize(PyObject) === Base.SizeUnknown()
        @test Base.eltype(list) === PyObject
        @test @pyv `eq($(list[0]), 1)`::Bool
        @test (list[0] = 99) === 99
        @test @pyv `eq($(list[0]), 99)`::Bool
        @test length(list) == 3
        @test delete!(list, 0) === list
        @test length(list) == 2
        @test @pyv `eq($list, [2,3])`::Bool
        clist = collect(list)
        @test length(clist) == 2
        @test @pyv `eq($(clist[1]), 2)`::Bool
        @test @pyv `eq($(clist[2]), 3)`::Bool
        @test hash(x) === UInt(1)
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
