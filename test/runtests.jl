using Python, Test, Dates, Compat

mutable struct Struct1
    x :: String
    y :: Int
end

@testset "Python.jl" begin

    @testset "cpython" begin
    end

    @testset "eval" begin
        @pyg ```
        import sys, os, datetime, array, io, fractions, array, ctypes
        eq = lambda a, b: type(a) is type(b) and a == b
        class Foo:
            def __init__(self, x=None):
                self.x = x
            def __repr__(self):
                return "Foo()"
            def __str__(self):
                return "<some Foo>"
        $(x::Int) = 234
        ```
        @test x === 234
        @test (@pyv `x`::Int x=123) == 123
        @test (@pyv `"$$"`::String) == "\$"
        @test (@pya `ans=1.0`::Float64) == 1.0
        @test (@pyr `return 12`::Int) == 12
    end

    @testset "convert-to-python" begin
        for v in [nothing, missing]
            @test @pyv `eq($v, None)`::Bool
        end
        @test @pyv `eq($true, True)`::Bool
        @test @pyv `eq($false, False)`::Bool
        for T in [Int8,Int16,Int32,Int64,Int128,UInt8,UInt16,UInt32,UInt64,UInt128,BigInt]
            @test @pyv `eq($(zero(T)), 0)`::Bool
            @test @pyv `eq($(Rational{T}(2,3)), fractions.Fraction(2,3))`::Bool
            @test @pyv `eq($(T(0):T(9)), range(10))`::Bool
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
        # PRIORITY=1000: wrapped julia values are just unwrapped
        for x in [nothing, missing, 2, 2.1, [], Int[1,2,3], (1,2,3), "foo", (x=1, y="two")]
            @test pyconvert(Any, pyjl(x)) === x
            @test pyconvert(typeof(x), pyjl(x)) === x
            @test pyconvert(Union{typeof(x), Nothing}, pyjl(x)) === x
        end

        # PRIORITY=200: buffer/array -> PyArray
        x = @pyv `b"foo"`::Any
        @test x isa PyVector{UInt8,UInt8,false,true}
        @test x == Vector{UInt8}("foo")
        x = @pyv `bytearray(b"foo")`::Any
        @test x isa PyVector{UInt8,UInt8,true,true}
        @test x == Vector{UInt8}("foo")
        x = @pyv `array.array("f", range(10))`::Any
        @test x isa PyVector{Float32,Float32,true,true}
        @test x == 0:9
        x = pyconvert(PyArray, [1 2; 3 4])
        @test x isa PyMatrix{Int,Int,true,true}
        @test x == [1 2; 3 4]

        # PRIORITY=100: other canonical conversions
        # None -> nothing
        x = @pyv `None`::Any
        @test x === nothing
        # bool -> Bool
        x = @pyv `True`::Any
        @test x === true
        # float -> Float64
        x = @pyv `1.5`::Any
        @test x === 1.5
        # complex -> Complex{Float64}
        x = @pyv `complex(1, 2.5)`::Any
        @test x === Complex(1, 2.5)
        # range -> StepRange
        x = @pyv `range(10)`::Any
        @test x === Clonglong(0):Clonglong(1):Clonglong(9)
        x = @pyv `range(2**1000)`::Any
        @test x isa StepRange{BigInt,Clonglong}
        @test x == 0:1:(big"2"^1000-1)
        x = @pyv `range(2**500, 2**1000, 2**400)`::Any
        @test x isa StepRange{BigInt,BigInt}
        @test x == (big"2"^500):(big"2"^400):(big"2"^1000-1)
        # str -> String
        x = @pyv `"hello there"`::Any
        @test x === "hello there"
        # tuple -> Tuple
        x = @pyv `(1,None,"foo")`::Any
        @test x === (Clonglong(1), nothing, "foo")
        x = @pyv `("foo",1,2,3)`::Tuple{String,Vararg{Int}}
        @test x === ("foo",1,2,3)
        x = @pyv `(None, 1)`::Tuple{Nothing,Int}
        @test x === (nothing, 1)
        # Mapping -> PyDict{PyObject, PyObject}
        x = @pyv `dict(x=1, y=2)`::Any
        @test x isa PyDict{PyObject, PyObject}
        # Sequence -> PyList{PyObject}
        x = @pyv `[1,2,3]`::Any
        @test x isa PyList{PyObject}
        # Set -> PySet{PyObject}
        x = @pyv `{1,2,3}`::Any
        @test x isa PySet{PyObject}
        x = @pyv `frozenset([1,2,3])`::Any
        @test x isa PySet{PyObject}
        # date -> Date
        x = @pyv `datetime.date(2001, 2, 3)`::Any
        @test x === Date(2001, 2, 3)
        # time -> Time
        x = @pyv `datetime.time(1, 2, 3, 4)`::Any
        @test x === Time(1, 2, 3, 0, 4)
        # datetime -> DateTime
        x = @pyv `datetime.datetime(2001, 2, 3, 4, 5, 6, 7000)`::Any
        @test x === DateTime(2001, 2, 3, 4, 5, 6, 7)
        # timedelta -> Period (Microsecond, unless overflow then Millisecond or Second)
        x = @pyv `datetime.timedelta(microseconds=12)`::Any
        @test x === Microsecond(12)
        # x = @pyv `datetime.timedelta(milliseconds=$(cld(typemax(Int),1000)+10))`::Any
        # @test x === Millisecond(cld(typemax(Int), 1000) + 10)
        # In fact, you can't make a timedelta big enough to overflow into seconds.
        # x = @pyv `datetime.timedelta(seconds=$(cld(typemax(Int),1000)+10))`::Any
        # @test x === Second(cld(typemax(Int), 1000) + 10)
        # Integral -> Integer (Clonglong, unless overflow then BigInt)
        x = @pyv `123`::Any
        @test x === Clonglong(123)
        x = @pyv `2**123`::Any
        @test x isa BigInt
        @test x == big"2"^123
        # TODO: numpy (e.g. numpy.float32 -> Float32)

        # PRIORITY=0: other reasonable conversions
        # None -> missing
        x = @pyv `None`::Missing
        @test x === missing
        x = @pyv `None`::Union{PyObject,Missing}
        @test x === missing
        # bytes -> Vector{UInt8}, Vector{Int8}, String
        x = @pyv `b"abc"`::Vector
        @test x isa Vector{UInt8}
        @test x == Vector{UInt8}("abc")
        x = @pyv `b"abc"`::Vector{Int8}
        @test x isa Vector{Int8}
        x = @pyv `b"abc"`::AbstractString
        @test x isa String
        @test x == "abc"
        # str -> Symbol, Char, Vector{UInt8}, Vector{Int8}
        x = @pyv `"foo"`::Symbol
        @test x === :foo
        x = @pyv `"x"`::AbstractChar
        @test x == 'x'
        x = @pyv `"abc"`::Vector{UInt8}
        @test x isa Vector{UInt8}
        @test x == Vector{UInt8}("abc")
        x = @pyv `"abc"`::Vector{Int8}
        @test x isa Vector{Int8}
        # range -> UnitRange
        x = @pyv `range(123)`::UnitRange{Int}
        @test x === 0:122
        # Iterable -> Vector, Set, Tuple, Pair
        x = @pyv `(1,2,3)`::Vector{Int}
        @test x isa Vector{Int}
        @test x == [1,2,3]
        x = @pyv `(1,2,3)`::Set{Int}
        @test x isa Set{Int}
        @test x == Set([1,2,3])
        x = @pyv `[1,2,3]`::Tuple{Int,Int,Int}
        @test x isa Tuple{Int,Int,Int}
        @test x === (1,2,3)
        x = @pyv `[4,5]`::Pair{Int,Int}
        @test x isa Pair{Int,Int}
        @test x === (4 => 5)
        # Mapping -> Dict
        x = @pyv `dict(x=1, y=2)`::Dict{String,Int}
        @test x isa Dict{String,Int}
        @test x == Dict("x"=>1, "y"=>2)
        # timedelta -> CompoundPeriod
        x = @pyv `datetime.timedelta(microseconds=123)`::Dates.CompoundPeriod
        @test x isa Dates.CompoundPeriod
        @test x == Dates.CompoundPeriod([Microsecond(123)])
        # Integral -> Rational, Real, Number, Any
        x = @pyv `123`::Rational{Int}
        @test x === 123//1
        x = @pyv `123`::AbstractFloat
        @test x === 123.0
        x = @pyv `123`::Complex{Int}
        @test x === Complex(123)
        # Real -> AbstractFloat, Number
        x = @pyv `1234.5`::AbstractFloat
        @test x === 1234.5
        x = @pyv `1234.5`::BigFloat
        @test x isa BigFloat
        @test x == big"1234.5"
        x = @pyv `1234.5`::Complex
        @test x === Complex(1234.5)
        x = @pyv `123.0`::Integer
        @test x === 123
        x = @pyv `1234.5`::Rational{Int}
        @test x === 2469//2
        # Complex -> Complex, Number, Any
        x = @pyv `complex(1,2)`::Complex
        @test x === Complex(1.0, 2.0)
        x = @pyv `complex(1,2)`::Complex{Int}
        @test x === Complex(1,2)
        x = @pyv `complex(1234.5, 0)`::AbstractFloat
        @test x === 1234.5
        x = @pyv `complex(123, 0)`::Integer
        @test x === 123
        # ctypes
        x = @pyv `ctypes.c_float(12)`::Number
        @test x === Cfloat(12)
        x = @pyv `ctypes.c_double(12)`::Number
        @test x === Cdouble(12)
        x = @pyv `ctypes.c_byte(12)`::Number
        @test x == Cchar(12)
        x = @pyv `ctypes.c_short(12)`::Number
        @test x === Cshort(12)
        x = @pyv `ctypes.c_int(12)`::Number
        @test x === Cint(12)
        x = @pyv `ctypes.c_long(12)`::Number
        @test x === Clong(12)
        x = @pyv `ctypes.c_longlong(12)`::Number
        @test x === Clonglong(12)
        x = @pyv `ctypes.c_ubyte(12)`::Number
        @test x == Cuchar(12)
        x = @pyv `ctypes.c_ushort(12)`::Number
        @test x === Cushort(12)
        x = @pyv `ctypes.c_uint(12)`::Number
        @test x === Cuint(12)
        x = @pyv `ctypes.c_ulong(12)`::Number
        @test x === Culong(12)
        x = @pyv `ctypes.c_ulonglong(12)`::Number
        @test x === Culonglong(12)
        x = @pyv `ctypes.c_char(12)`::Number
        @test x === Cchar(12)
        x = @pyv `ctypes.c_wchar('x')`::Number
        @test x === Cwchar_t('x')
        x = @pyv `ctypes.c_size_t(12)`::Number
        @test x === Csize_t(12)
        x = @pyv `ctypes.c_ssize_t(12)`::Number
        @test x === Cssize_t(12)
        x = @pyv `ctypes.c_void_p()`::Ptr
        @test x === C_NULL
        x = @pyv `ctypes.c_char_p()`::Ptr
        @test x === Ptr{Cchar}(0)
        x = @pyv `ctypes.c_char_p()`::Cstring
        @test x == Cstring(C_NULL)
        x = @pyv `ctypes.c_wchar_p()`::Ptr
        @test x === Ptr{Cwchar_t}(0)
        x = @pyv `ctypes.c_wchar_p()`::Cwstring
        @test x === Cwstring(C_NULL)
        # TODO: numpy

        # PRIORITY=-100: fallback to object -> PyObject
        x = @pyv `Foo()`::Any
        @test x isa PyObject
        x = @pyv `123`::PyObject
        @test x isa PyObject
        x = @pyv `123`::Union{Nothing,PyObject}
        @test x isa PyObject

        # PRIORITY=-200: conversions that must be specifically requested by excluding PyObject
        # object -> PyRef
        x = @pyv `123`::Union{Nothing,PyRef}
        @test x isa PyRef
        # buffer -> PyBuffer
        x = @pyv `b"abc"`::Union{Nothing,PyBuffer}
        @test x isa PyBuffer
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
        @test @pyv `eq($(pyfloat()), 0.0)`::Bool
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
        @test_throws PyException pymatmul(1,2)
        @test_throws PyException pyimatmul(1,2)
        @test @pyv `eq($(pyfloordiv(1,2)), 0)`::Bool
        @test @pyv `eq($(pyifloordiv(1,2)), 0)`::Bool
        @test @pyv `eq($(pytruediv(1,2)), 0.5)`::Bool
        @test @pyv `eq($(pyitruediv(1.0,2)), 0.5)`::Bool
        @test @pyv `eq($(pymod(1,2)), 1)`::Bool
        @test @pyv `eq($(pyimod(1,2)), 1)`::Bool
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
        @test @pyv `eq($(pyabs(-2)), 2)`::Bool
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
        z = PyObject(2)
        foo = @pyv `Foo()`
        Foo = @pyv `Foo`
        list = @pyv `[1,2,3]`
        dict = @pyv `{"x":1, "y":2}`
        arr = @pyv `array.array('f', [1,2,3])`
        mat = PyObject([1 2; 3 4])
        bio = @pyv `io.BytesIO()`
        @test x isa PyObject
        @test y isa PyObject
        @test foo isa PyObject
        @test Python.pyptr(y) === Python.C.PyNULL
        Python.C.PyErr_Clear()
        @test string(x) == "1"
        @test (io=IOBuffer(); print(io, x); String(take!(io))) == "1"
        @test repr(x) == "<py 1>"
        @test startswith(repr(foo), "<py Foo")
        @test (io=IOBuffer(); show(io, x); String(take!(io))) == "<py 1>"
        @test (io=IOBuffer(); ioc=IOContext(io, :typeinfo=>PyObject); show(ioc, x); String(take!(io))) == "1"
        @test (io=IOBuffer(); show(io, MIME("text/plain"), x); String(take!(io))) == "py: 1"
        @test @pyv `$(foo.x) is $foo.x`::Bool
        @test (foo.x = 99) === 99
        @test @pyv `eq($(foo.x), 99)`::Bool
        @test !hasproperty(x, :invalid)
        @test hasproperty(x, :__str__)
        @test issubset([:__str__, :__repr__, :x], propertynames(foo))
        @test x.jl!(Int8) === Int8(1)
        @test x.jl!i === Int(1)
        @test x.jl!b === true
        @test x.jl!s == "1"
        @test x.jl!r == "1"
        @test x.jl!f === Cdouble(1)
        @test x.jl!c === Complex{Cdouble}(1)
        @test collect(list.jl!iter(Int)) == [1,2,3]
        @test list.jl!list() isa PyList{PyObject}
        @test list.jl!list(Int) == [1,2,3]
        @test list.jl!set() isa PySet{PyObject}
        @test list.jl!set(Int) == Set([1,2,3])
        @test dict.jl!dict() isa PyDict{PyObject, PyObject}
        @test dict.jl!dict(String) isa PyDict{String, PyObject}
        @test dict.jl!dict(String, Int) == Dict("x"=>1, "y"=>2)
        @test arr.jl!buffer() isa PyBuffer
        @test arr.jl!array() == [1,2,3]
        @test arr.jl!array(Cfloat) == [1,2,3]
        @test arr.jl!array(Cfloat, 1) == [1,2,3]
        @test arr.jl!array(Cfloat, 1, Cfloat) == [1,2,3]
        @test arr.jl!array(Cfloat, 1, Cfloat, true) == [1,2,3]
        @test arr.jl!array(Cfloat, 1, Cfloat, true, true) == [1,2,3]
        @test arr.jl!vector() == [1,2,3]
        @test arr.jl!vector(Cfloat) == [1,2,3]
        @test arr.jl!vector(Cfloat, Cfloat) == [1,2,3]
        @test arr.jl!vector(Cfloat, Cfloat, true) == [1,2,3]
        @test arr.jl!vector(Cfloat, Cfloat, true, true) == [1,2,3]
        @test mat.jl!matrix() == [1 2; 3 4]
        @test mat.jl!matrix(Int) == [1 2; 3 4]
        @test mat.jl!matrix(Int, Int) == [1 2; 3 4]
        @test mat.jl!matrix(Int, Int, true) == [1 2; 3 4]
        @test mat.jl!matrix(Int, Int, true, false) == [1 2; 3 4]
        @test bio.jl!io() isa PyIO
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
        # comparison
        @test @pyv `$(x == x) is True`::Bool
        @test @pyv `$(x == z) is False`::Bool
        @test @pyv `$(x != x) is False`::Bool
        @test @pyv `$(x != z) is True`::Bool
        @test @pyv `$(x <= x) is True`::Bool
        @test @pyv `$(x <= z) is True`::Bool
        @test @pyv `$(x < x) is False`::Bool
        @test @pyv `$(x < z) is True`::Bool
        @test @pyv `$(x >= x) is True`::Bool
        @test @pyv `$(x >= z) is False`::Bool
        @test @pyv `$(x > x) is False`::Bool
        @test @pyv `$(x > z) is False`::Bool
        @test isequal(x, x)
        @test !isequal(x, z)
        @test !isless(x, x)
        @test isless(x, z)
        # arithmetic
        @test @pyv `eq($(zero(PyObject)), 0)`::Bool
        @test @pyv `eq($(one(PyObject)), 1)`::Bool
        @test @pyv `eq($(-x), -1)`::Bool
        @test @pyv `eq($(+x), 1)`::Bool
        @test @pyv `eq($(~x), ~1)`::Bool
        @test @pyv `eq($(abs(x)), 1)`::Bool
        @test @pyv `eq($(x+z), 3)`::Bool
        @test @pyv `eq($(x-z), -1)`::Bool
        @test @pyv `eq($(x*z), 2)`::Bool
        @test @pyv `eq($(x/z), 0.5)`::Bool
        @test @pyv `eq($(fld(x,z)), 0)`::Bool
        @test @pyv `eq($(mod(x,z)), 1)`::Bool
        @test @pyv `eq($(z^z), 4)`::Bool
        @test @pyv `eq($(x<<z), 4)`::Bool
        @test @pyv `eq($(z>>x), 1)`::Bool
        @test @pyv `eq($(x&z), 0)`::Bool
        @test @pyv `eq($(x|z), 3)`::Bool
        @test @pyv `eq($(xor(x,z)), 3)`::Bool
        @test @pyv `eq($(x+2), 3)`::Bool
        @test @pyv `eq($(x-2), -1)`::Bool
        @test @pyv `eq($(x*2), 2)`::Bool
        @test @pyv `eq($(x/2), 0.5)`::Bool
        @test @pyv `eq($(fld(x,2)), 0)`::Bool
        @test @pyv `eq($(mod(x,2)), 1)`::Bool
        @test @pyv `eq($(z^2), 4)`::Bool
        @test @pyv `eq($(x<<2), 4)`::Bool
        @test @pyv `eq($(z>>1), 1)`::Bool
        @test @pyv `eq($(x&2), 0)`::Bool
        @test @pyv `eq($(x|2), 3)`::Bool
        @test @pyv `eq($(xor(x,2)), 3)`::Bool
        @test @pyv `eq($(1+z), 3)`::Bool
        @test @pyv `eq($(1-z), -1)`::Bool
        @test @pyv `eq($(1*z), 2)`::Bool
        @test @pyv `eq($(1/z), 0.5)`::Bool
        @test @pyv `eq($(fld(1,z)), 0)`::Bool
        @test @pyv `eq($(mod(1,z)), 1)`::Bool
        # @test @pyv `eq($(2^z), 4)`::Bool
        # @test @pyv `eq($(1<<z), 4)`::Bool
        # @test @pyv `eq($(2>>x), 1)`::Bool
        @test @pyv `eq($(1&z), 0)`::Bool
        @test @pyv `eq($(1|z), 3)`::Bool
        @test @pyv `eq($(xor(1,z)), 3)`::Bool
        @test @pyv `eq($(powermod(z,3,5)), 3)`::Bool
    end

    @testset "PyDict" begin
        o = pydict(a=1, b=2)
        @test PyDict(o) isa PyDict{PyObject, PyObject}
        @test PyDict{String}(o) isa PyDict{String, PyObject}
        @test PyDict{String, Int}(o) isa PyDict{String, Int}
        d = PyDict{String, Int}(o)
        d2 = copy(d)
        @test d == Dict("a"=>1, "b"=>2)
        @test Set(keys(d)) == Set(["a", "b"])
        @test Set(values(d)) == Set([1, 2])
        @test length(d) == 2
        @test d["a"] == 1
        @test d["b"] == 2
        @test d2 == d
        d["c"] = 3
        @test d["c"] == 3
        @test length(d) == 3
        @test haskey(d, "c")
        @test d2 != d
        @test !haskey(d2, "c")
        delete!(d, "c")
        @test length(d) == 2
        @test !haskey(d, "c")
        empty!(d2)
        @test length(d2) == 0
        @test !haskey(d2, "a")
        @test !haskey(d2, "b")
        @test get(d, "a", 0) == 1
        @test get(d, "x", 0) == 0
        @test !haskey(d, "x")
        @test get(()->0, d, "a") == 1
        @test get(()->0, d, "x") == 0
        @test !haskey(d, "x")
        @test get!(d, "a", 0) == 1
        @test length(d) == 2
        @test get!(d, "x", 0) == 0
        @test length(d) == 3
        @test d["x"] == 0
        delete!(d, "x")
        @test length(d) == 2
        @test !haskey(d, "x")
        @test get!(()->0, d, "a") == 1
        @test length(d) == 2
        @test get!(()->0, d, "x") == 0
        @test length(d) == 3
        @test d["x"] == 0
    end

    @testset "PyIO" begin
        bio = PyIO((@pyv `io.BytesIO()`), buflen=1)
        sio = PyIO((@pyv `io.StringIO()`), buflen=1)
        @test bio isa PyIO
        @test sio isa PyIO
        @test !bio.text
        @test sio.text
        @test flush(bio) === nothing
        @test flush(sio) === nothing
        @test eof(bio)
        @test eof(sio)
        @test eof(bio)
        @test eof(sio)
        @test_throws PyException fd(bio)
        @test_throws PyException fd(sio)
        @test isreadable(bio)
        @test isreadable(sio)
        @test iswritable(bio)
        @test iswritable(sio)
        write(bio, "hello")
        write(sio, "foo")
        flush(bio)
        flush(sio)
        seekstart(bio)
        seekstart(sio)
        bpos = position(bio)
        spos = position(sio)
        @test bpos isa Int
        @test spos isa Int
        @test !eof(bio)
        @test !eof(sio)
        seekend(bio)
        seekend(sio)
        @test eof(bio)
        @test eof(sio)
        seek(bio, bpos)
        seek(sio, bpos)
        @test position(bio) == bpos
        @test position(sio) == spos
        skip(bio, 2)
        skip(sio, 1)
        skip(bio, -1)
        @test_throws Exception skip(sio, -1)
        read(sio, length(sio.ibuf))
        @test position(bio) > bpos
        @test position(sio) > spos
        seekstart(bio)
        seekstart(sio)
        @test read(bio, String) == "hello"
        @test read(sio, String) == "foo"
        @test eof(bio)
        @test eof(sio)
        bpos = position(bio)
        spos = position(sio)
        @test write(bio, 'x') == 1
        @test write(sio, 'x') == 1
        seekstart(bio)
        seekstart(sio)
        @test read(bio, String) == "hellox"
        @test read(sio, String) == "foox"
        truncate(bio, bpos)
        truncate(sio, spos)
        seekstart(bio)
        seekstart(sio)
        @test read(bio, String) == "hello"
        @test read(sio, String) == "foo"
        @test isopen(bio)
        @test isopen(sio)
        @test close(bio) === nothing
        @test close(sio) === nothing
        @test !isopen(bio)
        @test !isopen(sio)
    end

    @testset "PyArray" begin
        vec = [1, 2, 3]
        mat = transpose([1.0 2.5; 3.0 4.5]) # transpose => not linearly indexable
        arr = PyObjectArray(reshape(1:8, 2, 2, 2))
        @test_throws Exception PyArray(pylist())
        @test PyArray(vec) isa PyVector{Int, Int, true, true}
        @test PyVector(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int}(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int,1}(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int,1,Int}(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int,1,Int,true}(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int,1,Int,true,true}(vec) isa PyVector{Int, Int, true, true}
        @test PyArray{Int,1,Int,false,false}(vec) isa PyVector{Int, Int, false, false}
        @test PyArray(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyMatrix(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray{Float64}(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray{Float64,2}(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray{Float64,2,Float64}(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray{Float64,2,Float64,true}(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray{Float64,2,Float64,true,false}(mat) isa PyMatrix{Float64, Float64, true, false}
        @test PyArray(arr) isa PyArray{PyObject, 3, Python.CPython.PyObjectRef, true, true}
        @test PyArray(["foo", "bar"]) isa PyVector{PyObject, Python.CPython.PyObjectRef, true, true}
        veco = PyArray(vec)
        mato = PyArray(mat)
        arro = PyArray(arr)
        @test Python.ismutablearray(veco)
        @test Python.ismutablearray(mato)
        @test Python.ismutablearray(arro)
        @test !Python.ismutablearray(PyArray{Int,1,Int,false,true}(vec))
        @test size(veco) == size(vec)
        @test size(mato) == size(mat)
        @test size(arro) == size(arr)
        @test length(veco) == length(vec)
        @test length(mato) == length(mat)
        @test length(arro) == length(arr)
        @test isequal(veco, vec)
        @test isequal(mato, mat)
        @test isequal(arro, arr)
        @test Base.IndexStyle(typeof(veco)) == Base.IndexLinear()
        @test Base.IndexStyle(typeof(mato)) == Base.IndexCartesian()
        @test Base.IndexStyle(typeof(arro)) == Base.IndexLinear()
        vec[1] += 1
        mat[1] += 1
        arr[1] += 1
        @test isequal(veco, vec)
        @test isequal(mato, mat)
        @test isequal(arro, arr)
        veco[1] += 1
        mato[1] += 1
        arro[1] += 1
        @test isequal(veco, vec)
        @test isequal(mato, mat)
        @test isequal(arro, arr)
        mato[2,2] += 1
        @test isequal(mato, mat)
        arro[2,2,2] += 1
        @test isequal(arro, arr)
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
            @test @pyv `repr($(pyjl(missing))) == "jl: missing"`::Bool
            @test @pyv `str($(pyjl(missing))) == "missing"`::Bool
            x = Struct1("foo", 2)
            @test @pyv `$(pyjl(x)).x == "foo"`::Bool
            @test @pyv `$(pyjl(x)).y == 2`::Bool
            @py `$(pyjl(x)).y = 0`
            @test x.y == 0
            @test @pyv `"x" in dir($(pyjl(x)))`::Bool
            @test @pyv `"Int" in dir($(pyjl(Base)))`::Bool
            @test @pyv `"push_b" in dir($(pyjl(Base)))`::Bool
            @test @pyv `$(Int)(1.0) == 1`::Bool
            @test (@pyv `$(pyjl(Vector))()`::Any) == Vector()
            @test (@pyv `$(pyjl(sort))($(pyjl([1,2,3])), rev=True)`::Any) == [3,2,1]
            @test_throws PyException @pyv `$(pyjl(nothing))("not", "callable")`::Union{}
            @test @pyv `len($(pyjl((1,2,3)))) == 3`::Bool
            @test_throws PyException @pyv `len($(pyjl(x)))`::Bool
            x = Dict(1=>2)
            @test @pyv `$(pyjl(x))[1] == 2`::Bool
            @test_throws PyException @pyv `$(pyjl((1, 2)))[0]`::Union{}
            @test_throws PyException @pyv `$(pyjl(Dict("x"=>1, "y"=>2)))["z"]`::Union{}
            @py `$(pyjl(x))[1] = 0`
            @test x[1] == 0
            @py `del $(pyjl(x))[1]`
            @test isempty(x)
            @test_throws PyException @py `$(pyjl((1, 2)))[1] = 0`::Union{}
            @test_throws PyException @py `$(pyjl(Dict("x"=>1, "y"=>2)))[nothing] = 0`::Union{}
            @test @pyv `1 in $(pyjl((1,2,3)))`::Bool
            @test @pyv `0 not in $(pyjl((1,2,3)))`::Bool
            @test @pyv `1.5 not in $(pyjl((1,2,3)))`::Bool
            @test @pyv `$(pyjl(3)) == 3`::Bool
            @test @pyv `not ($(pyjl(3)) == 5)`::Bool
            @test @pyv `not ($(pyjl(3)) != 3)`::Bool
            @test @pyv `$(pyjl(3)) != 5`::Bool
            @test @pyv `$(pyjl(3)) <= 3`::Bool
            @test @pyv `not ($(pyjl(3)) <= 2)`::Bool
            @test @pyv `not ($(pyjl(3)) < 3)`::Bool
            @test @pyv `$(pyjl(3)) < 5`::Bool
            @test @pyv `$(pyjl(3)) >= 3`::Bool
            @test @pyv `not ($(pyjl(3)) >= 5)`::Bool
            @test @pyv `not ($(pyjl(3)) > 3)`::Bool
            @test @pyv `$(pyjl(3)) > 2`::Bool
            @test @pyv `$(pyjl(identity)).__name__ == "identity"`::Bool
        end

        @testset "juliaio" begin
            for value in Any[stdin, stdout, IOBuffer()]
                @test @pyv `type($(pyjl(value))).__name__ == "BufferedIOValue"`::Bool
                @test @pyv `type($(pyrawio(value))).__name__ == "RawIOValue"`::Bool
                @test @pyv `type($(pybufferedio(value))).__name__ == "BufferedIOValue"`::Bool
                @test @pyv `type($(pytextio(value))).__name__ == "TextIOValue"`::Bool
            end
            io = IOBuffer()
            @test @pyv `"BufferedIOValue" in $io.__class__.__name__`::Bool
            @test @pyv `"BufferedIOValue" in $(pybufferedio(io)).__class__.__name__`::Bool
            @test @pyv `"BufferedIOValue" in $io.tobufferedio().__class__.__name__`::Bool
            @test @pyv `"RawIOValue" in $io.torawio().__class__.__name__`::Bool
            @test @pyv `"TextIOValue" in $io.totextio().__class__.__name__`::Bool
            @test @pyv `"TextIOValue" in $(pytextio(io)).__class__.__name__`::Bool
            @test @pyv `"TextIOValue" in $(pytextio(io)).totextio().__class__.__name__`::Bool
            @test @pyv `"BufferedIOValue" in $(pytextio(io)).tobufferedio().__class__.__name__`::Bool
            @test @pyv `"RawIOValue" in $(pytextio(io)).torawio().__class__.__name__`::Bool
            @test @pyv `"RawIOValue" in $(pyrawio(io)).__class__.__name__`::Bool
            @test @pyv `"RawIOValue" in $(pyrawio(io)).torawio().__class__.__name__`::Bool
            @test @pyv `"BufferedIOValue" in $(pyrawio(io)).tobufferedio().__class__.__name__`::Bool
            @test @pyv `"TextIOValue" in $(pyrawio(io)).totextio().__class__.__name__`::Bool
            println(io, "foo")
            println(io, "bar")
            seekstart(io)
            @test @pyv `eq(list($io), [b"foo\n", b"bar\n"])`::Bool
            @test @pyv `not $io.closed`::Bool
            @test_throws PyException @pyv `$io.fileno()`::Int
            @test @pyv `$io.flush() is None`::Bool
            @test @pyv `not $io.isatty()`::Bool
            @test @pyv `$io.readable()`::Bool
            @test @pyv `$io.writable()`::Bool
            @test @pyv `$io.close() is None`::Bool
            @test @pyv `$io.closed`::Bool
            io = IOBuffer()
            @test @pyv `$io.tell() == 0`::Bool
            print(io, "foo")
            @test @pyv `$io.tell() == 3`::Bool
            @test @pyv `$io.seek(2) == 2`::Bool
            @test @pyv `$io.seek(1, 0) == 1`::Bool
            @test @pyv `$io.seek(1, 1) == 2`::Bool
            @test @pyv `$io.seek(-2, 2) == 1`::Bool
            @test_throws PyException @pyv `$io.seek(0, 3)`::Union{}
            @test @pyv `$io.seek(2) == 2`::Bool
            @test @pyv `$io.truncate() == 2`::Bool
            @test @pyv `$io.truncate(1) == 1`::Bool
            @test @pyv `$io.tell() == 1`::Bool
            @test @pyv `$io.seekable()`::Bool
            io = IOBuffer()
            @test @pyv `$io.writelines([b"f", b"oo"]) is None`::Bool
            @test String(take!(io)) == "foo"
            println(io, "foo")
            println(io, "bar")
            seekstart(io)
            @test @pyv `$io.readlines() == [b"foo\n", b"bar\n"]`::Bool
            @test @pyv `eq($(pytextio(io)).encoding, "UTF-8")`::Bool
            @test @pyv `eq($(pytextio(io)).errors, "strict")`::Bool
            @test_throws PyException @pyv `$(pytextio(io)).detach()`::Union{}
            io = IOBuffer()
            @test @pyv `$io.write(b"fooo") == 4`::Bool
            @test String(take!(io)) == "fooo"
            @test @pyv `$io.totextio().write("a\nb\nc") == 5`::Bool
            linesep = @pyv `os.linesep`::String
            @test String(take!(io)) == "a$(linesep)b$(linesep)c"
            x = zeros(UInt8, 10)
            io = IOBuffer()
            write(io, "foo")
            seekstart(io)
            @test @pyv `$io.readinto($x) == 3`::Bool
            @test String(take!(io)) == "foo"
            write(io, "bar")
            seekstart(io)
            @test @pyv `eq($io.read(), b"bar")`::Bool
            seekstart(io)
            @test @pyv `eq($io.totextio().read(), "bar")`::Bool
            io = IOBuffer()
            println(io, "foo")
            println(io, "bar")
            seekstart(io)
            @test @pyv `eq($io.readline(), b"foo\n")`::Bool
            @test @pyv `eq($io.totextio().readline(), "bar\n")`::Bool
        end
    end

end
