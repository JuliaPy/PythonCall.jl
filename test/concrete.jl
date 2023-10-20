@testitem "import" begin
    sys = pyimport("sys")
    os = pyimport("os")
    @test pyeq(Bool, sys.__name__, "sys")
    @test pyeq(Bool, os.__name__, "os")
    sysos = pyimport("sys", "os")
    @test sysos isa Tuple{Py, Py}
    @test pyis(sysos[1], sys)
    @test pyis(sysos[2], os)
    ver = pyimport("sys" => "version")
    @test pyis(ver, sys.version)
    path = pyimport("sys" => "path")
    @test pyis(path, sys.path)
    verpath = pyimport("sys" => ("version", "path"))
    @test verpath isa Tuple{Py, Py}
    @test pyis(verpath[1], ver)
    @test pyis(verpath[2], path)
end

@testitem "consts" begin
    @test pybuiltins.None isa Py
    @test pystr(String, pybuiltins.None) == "None"
end

@testitem "str" begin
    @test pyisinstance(pystr("foo"), pybuiltins.str)
    @test pyeq(Bool, pystr(pystr("foo")), pystr("foo"))
    @test pyeq(Bool, pystr(SubString("foobarbaz", 4:6)), pystr("bar"))
    @test pyeq(Bool, pystr('x'), pystr("x"))
    @test pystr(String, pybuiltins.None) === "None"
    @test pystr(String, pyint(123)) === "123"
    @test pystr(String, pystr("foo")) === "foo"
end

@testitem "bytes" begin
    @test pyisinstance(pybytes(UInt8[1,2,3]), pybuiltins.bytes)
    @test pyeq(Bool, pybytes(pylist([1,2,3])), pybytes(UInt8[1,2,3]))
    @test pyeq(Bool, pybytes(b"foo"), pystr("foo").encode("ascii"))
    @test pyeq(Bool, pybytes(codeunits(SubString("foobarbaz", 4:6))), pystr("bar").encode("ascii"))
    @test pybytes(Vector, pylist([1,2,3])) == UInt8[1,2,3]
    @test pybytes(Vector{UInt8}, pylist([1,2,3])) == UInt8[1,2,3]
    @test pybytes(Base.CodeUnits, pystr("foo").encode("ascii")) == b"foo"
    @test pybytes(Base.CodeUnits{UInt8,String}, pystr("bar").encode("ascii")) == b"bar"
end

@testitem "tuple" begin
    z = pytuple()
    @test pyisinstance(z, pybuiltins.tuple)
    @test pylen(z) == 0
    x = pytuple((1,2,3))
    @test pyisinstance(x, pybuiltins.tuple)
    @test pylen(x) == 3
    @test pyeq(Bool, pygetitem(x, 0), 1)
    @test pyeq(Bool, pygetitem(x, 1), 2)
    @test pyeq(Bool, pygetitem(x, 2), 3)
    @test pyeq(Bool, pytuple([1,2,3]), x)
    @test pyeq(Bool, pytuple(i+1 for i in 0:10 if i<3), x)
    @test pyeq(Bool, pytuple(pytuple((1,2,3))), x)
    @test pyeq(Bool, pytuple(pylist([1,2,3])), x)
end

@testitem "list" begin
    z = pylist()
    @test pyisinstance(z, pybuiltins.list)
    @test pylen(z) == 0
    x = pylist((1,2,3))
    @test pyisinstance(x, pybuiltins.list)
    @test pylen(x) == 3
    @test pyeq(Bool, pygetitem(x, 0), 1)
    @test pyeq(Bool, pygetitem(x, 1), 2)
    @test pyeq(Bool, pygetitem(x, 2), 3)
    @test pyeq(Bool, pylist([1,2,3]), x)
    @test pyeq(Bool, pylist(i+1 for i in 0:10 if i<3), x)
    @test pyeq(Bool, pylist(pylist((1,2,3))), x)
    @test pyeq(Bool, pylist(pytuple([1,2,3])), x)
    @test pyeq(Bool, pycollist([1,2,3]), pylist([1,2,3]))
    @test pyeq(Bool, pycollist([1 2; 3 4]), pylist((pylist([1,3]), pylist([2,4]))))
    @test pyeq(Bool, pyrowlist([1,2,3]), pylist([1,2,3]))
    @test pyeq(Bool, pyrowlist([1 2; 3 4]), pylist((pylist([1,2]), pylist([3,4]))))
end

@testitem "dict" begin
    z = pydict()
    @test pyisinstance(z, pybuiltins.dict)
    @test pylen(z) == 0
    x = pydict(foo=1, bar=2)
    @test pyisinstance(x, pybuiltins.dict)
    @test pylen(x) == 2
    @test pyeq(Bool, pygetitem(x, "foo"), 1)
    @test pyeq(Bool, pygetitem(x, "bar"), 2)
    @test pyeq(Bool, pydict(["foo"=>1, "bar"=>2]), x)
    @test pyeq(Bool, pydict([("foo"=>1), ("bar"=>2)]), x)
    @test pyeq(Bool, pydict(Dict("foo"=>1, "bar"=>2)), x)
    @test pyeq(Bool, pydict((foo=1, bar=2)), x)
    @test pyeq(Bool, pydict(x), x)
end

@testitem "bool" begin
    @test pyis(pybool(), pybuiltins.False)
    @test pyis(pybool(false), pybuiltins.False)
    @test pyis(pybool(true), pybuiltins.True)
    @test pyis(pybool(0.0), pybuiltins.False)
    @test pyis(pybool(-1.2), pybuiltins.True)
    @test pyis(pybool(pybuiltins.None), pybuiltins.False)
    @test pyis(pybool(pylist()), pybuiltins.False)
    @test pyis(pybool(pylist([1,2,3])), pybuiltins.True)
end

@testitem "int" begin
    @test pyisinstance(pyint(), pybuiltins.int)
    @test pystr(String, pyint()) == "0"
    x = 123
    y = pyint(x)
    @test pyisinstance(y, pybuiltins.int)
    @test pystr(String, y) == string(x)
    x = BigInt(123) << 200
    y = pyint(x)
    @test pyisinstance(y, pybuiltins.int)
    @test pystr(String, y) == string(x)
    x = UInt(123)
    y = pyint(x)
    @test pyisinstance(y, pybuiltins.int)
    @test pystr(String, y) == string(x)
    x = UInt128(123) << 100
    y = pyint(x)
    @test pyisinstance(y, pybuiltins.int)
    @test pystr(String, y) == string(x)
    @test pyeq(Bool, pyint(pyint(123)), pyint(123))
    @test pyeq(Bool, pyint(pyfloat(12.3)), pyint(12))
end

@testitem "float" begin
    y = pyfloat()
    @test pyisinstance(y, pybuiltins.float)
    @test pyeq(Bool, y, pyint(0))
    x = 123
    y = pyfloat(x)
    @test pyisinstance(y, pybuiltins.float)
    @test pyeq(Bool, y, pyint(x))
    x = 0.25
    y = pyfloat(x)
    @test pyisinstance(y, pybuiltins.float)
    @test pyeq(Bool, y, pytruediv(1, 4))
    x = 1//4
    y = pyfloat(x)
    @test pyisinstance(y, pybuiltins.float)
    @test pyeq(Bool, y, pyfloat(float(x)))
    @test pyeq(Bool, pyfloat(pyfloat(12.3)), pyfloat(12.3))
    @test pyeq(Bool, pyfloat(pyint(123)), pyfloat(123))
end

@testitem "complex" begin
    y = pycomplex()
    @test pyisinstance(y, pybuiltins.complex)
    @test pyeq(Bool, y, pyint(0))
    x = 12.3
    y = pycomplex(x)
    @test pyisinstance(y, pybuiltins.complex)
    @test pyeq(Bool, y, pyfloat(x))
    xr, xi = 12, 34
    y = pycomplex(xr, xi)
    @test pyisinstance(y, pybuiltins.complex)
    @test pyeq(Bool, y.real, pyfloat(xr))
    @test pyeq(Bool, y.imag, pyfloat(xi))
    x = Complex(12, 34)
    y = pycomplex(x)
    @test pyisinstance(y, pybuiltins.complex)
    @test pyeq(Bool, y.real, pyfloat(real(x)))
    @test pyeq(Bool, y.imag, pyfloat(imag(x)))
    @test pyeq(Bool, pycomplex(y), y)
    @test pyeq(Bool, pycomplex(pyint(12), pyint(34)), y)
end

@testitem "set" begin
    y = pyset()
    yf = pyfrozenset()
    @test pyisinstance(y, pybuiltins.set)
    @test pylen(y) == 0
    @test pyisinstance(yf, pybuiltins.frozenset)
    @test pylen(yf) == 0
    @test pyeq(Bool, y, yf)
    x = [1,2,3,2,1]
    y = pyset(x)
    yf = pyfrozenset(x)
    @test pyisinstance(y, pybuiltins.set)
    @test pylen(y) == 3
    @test pycontains(y, 1)
    @test pycontains(y, 2)
    @test pycontains(y, 3)
    @test pyeq(Bool, pyset(y), y)
    @test pyisinstance(yf, pybuiltins.frozenset)
    @test pylen(yf) == 3
    @test pycontains(yf, 1)
    @test pycontains(yf, 2)
    @test pycontains(yf, 3)
    @test pyeq(Bool, pyfrozenset(y), y)
    @test pyeq(Bool, y, yf)
end

@testitem "slice" begin
    x = pyslice(12)
    @test pyisinstance(x, pybuiltins.slice)
    @test pyeq(Bool, x.start, pybuiltins.None)
    @test pyeq(Bool, x.stop, 12)
    @test pyeq(Bool, x.step, pybuiltins.None)
    x = pyslice(12, 34)
    @test pyisinstance(x, pybuiltins.slice)
    @test pyeq(Bool, x.start, 12)
    @test pyeq(Bool, x.stop, 34)
    @test pyeq(Bool, x.step, pybuiltins.None)
    x = pyslice(12, 34, 56)
    @test pyisinstance(x, pybuiltins.slice)
    @test pyeq(Bool, x.start, 12)
    @test pyeq(Bool, x.stop, 34)
    @test pyeq(Bool, x.step, 56)
end

@testitem "range" begin
    x = pyrange(123)
    @test pyisinstance(x, pybuiltins.range)
    @test pyeq(Bool, x.start, 0)
    @test pyeq(Bool, x.stop, 123)
    @test pyeq(Bool, x.step, 1)
    x = pyrange(12, 123)
    @test pyisinstance(x, pybuiltins.range)
    @test pyeq(Bool, x.start, 12)
    @test pyeq(Bool, x.stop, 123)
    @test pyeq(Bool, x.step, 1)
    x = pyrange(12, 123, 3)
    @test pyisinstance(x, pybuiltins.range)
    @test pyeq(Bool, x.start, 12)
    @test pyeq(Bool, x.stop, 123)
    @test pyeq(Bool, x.step, 3)
end

@testitem "none" begin
    # TODO
end

@testitem "type" begin
    x = pytype(pyint())
    @test pyisinstance(x, pybuiltins.type)
    @test pyis(x, pybuiltins.int)
    x = pytype(pybuiltins.type)
    @test pyisinstance(x, pybuiltins.type)
    @test pyis(x, pybuiltins.type)
    x = pytype("Foo", (), ["foo"=>1, "bar"=>2])
    @test pyisinstance(x, pybuiltins.type)
    @test pyeq(Bool, x.__name__, "Foo")
    @test pyeq(Bool, x.foo, 1)
    @test pyeq(Bool, x.bar, 2)
end

@testitem "fraction" begin
    # TODO
end

@testitem "method" begin
    # TODO
end

@testitem "datetime" begin
    using Dates
    dt = pyimport("datetime")
    x1 = pydate(2001, 2, 3)
    @test pyisinstance(x1, dt.date)
    @test pyeq(Bool, x1, dt.date(2001, 2, 3))
    x2 = pydate(Date(2002, 3, 4))
    @test pyisinstance(x2, dt.date)
    @test pyeq(Bool, x2, dt.date(2002, 3, 4))
    x3 = pytime(12, 3, 4, 5)
    @test pyisinstance(x3, dt.time)
    @test pyeq(Bool, x3, dt.time(12, 3, 4, 5))
    x4 = pytime(Time(23, 4, 5, 0, 6))
    @test pyisinstance(x4, dt.time)
    @test pyeq(Bool, x4, dt.time(23, 4, 5, 6))
    x5 = pydatetime(2001, 2, 3, 4, 5, 6, 7)
    @test pyisinstance(x5, dt.datetime)
    @test pyeq(Bool, x5, dt.datetime(2001, 2, 3, 4, 5, 6, 7))
    x6 = pydatetime(Date(2007, 8, 9))
    @test pyisinstance(x6, dt.datetime)
    @test pyeq(Bool, x6, dt.datetime(2007, 8, 9))
    x7 = pydatetime(DateTime(2001, 2, 3, 4, 5, 6, 7))
    @test pyisinstance(x7, dt.datetime)
    @test pyeq(Bool, x7, dt.datetime(2001, 2, 3, 4, 5, 6, 7000))
end

@testitem "code" begin
    # check for ArgumentError when inputs are mixed up
    @test_throws ArgumentError pyeval(Main, "1+1")
    @test_throws ArgumentError pyeval(Main, Main)
    @test_throws ArgumentError pyeval("1+1", "1+1")
    @test_throws ArgumentError pyexec(Main, "1+1")
    @test_throws ArgumentError pyexec(Main, Main)
    @test_throws ArgumentError pyexec("1+1", "1+1")
    # basic code execution
    m = Module(:test)
    g = pydict()
    @test pyeq(Bool, pyeval("1+1", m), 2)
    @test pyeq(Bool, pyeval("1+1", g), 2)
    @test pyeq(Bool, pyeval(pystr("1+1"), g), 2)
    @test pyexec("1+1", m) === nothing
    @test pyexec("1+1", g) === nothing
    @test pyexec(pystr("1+1"), g) === nothing
    # check the globals are what we think they are
    @test pyis(pyeval("globals()", g), g)
    mg = pyeval("globals()", m)
    @test pyisinstance(mg, pybuiltins.dict)
    # these automatically gain 1 item, __builtins__
    @test length(g) == 1
    @test length(mg) == 1
    @test pycontains(g, "__builtins__")
    @test pycontains(mg, "__builtins__")
    # code should fail when x does not exist
    @test_throws PyException pyeval("x+1", g)
    @test_throws PyException pyeval("x+1", g)
    # now set x and try again
    g["x"] = 1
    @test pyeq(Bool, pyeval("x+1", g), 2)
    # set x using pyexec this time
    pyexec("x=2", g)
    @test pyeq(Bool, g["x"], 2)
    @test pyeq(Bool, pyeval("x+1", g), 3)
    # now use locals
    # check empty locals have no effect
    l = pydict()
    @test pyeq(Bool, pyeval("x+1", g, l), 3)
    @test pyeq(Bool, pyeval("x+1", g, Dict()), 3)
    # now set x locally
    l["x"] = 3
    @test pyeq(Bool, pyeval("x+1", g, l), 4)
    @test pyeq(Bool, pyeval("x+1", g, Dict()), 3)
    @test pyeq(Bool, pyeval("x+1", g, Dict("x" => 0)), 1)
    @test pyeq(Bool, pyeval("x+1", g, (x=1,)), 2)
    # check pyexec runs in local scope
    pyexec("x=4", g, l)
    @test pyeq(Bool, g["x"], 2)
    @test pyeq(Bool, l["x"], 4)
    # check global code runs in global scope
    pyexec("global y; y=x+1", g, l)
    @test pyeq(Bool, g["y"], 5)
    @test !pycontains(l, "y")
    # check pyeval converts types correctly
    @test pyeval(Int, "1+1", g) === 2
    @test pyeval(Nothing, "None", g) === nothing
end
