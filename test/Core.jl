@testitem "object" begin
    import Markdown
    @testset "pyis" begin
        x = pylist()
        y = PythonCall.pynew(x)
        z = pylist()
        @test pyis(x, x)
        @test pyis(x, y)
        @test !pyis(x, z)
    end
    @testset "pyrepr" begin
        @test pyrepr(String, pybuiltins.None) == "None"
        @test pyrepr(String, pyint(123)) == "123"
        @test pyrepr(String, "hello") == "'hello'"
    end
    @testset "pyascii" begin
        @test pyascii(String, pybuiltins.None) == "None"
        @test pyascii(String, pyint(234)) == "234"
        @test pyascii(String, "hello") == "'hello'"
    end
    @testset "pyhasattr" begin
        @test pyhasattr(pybuiltins.None, "__class__")
        @test !pyhasattr(pybuiltins.None, "not_an_attr")
    end
    @testset "pygetattr" begin
        @test pyisinstance(pygetattr(pybuiltins.None, "__class__"), pybuiltins.type)
        @test_throws PyException pygetattr(pybuiltins.None, "not_an_attr")
        @test pyisinstance(
            pygetattr(pybuiltins.None, "__class__", pybuiltins.None),
            pybuiltins.type,
        )
        @test pyis(
            pygetattr(pybuiltins.None, "not_an_attr", pybuiltins.None),
            pybuiltins.None,
        )
    end
    @testset "pysetattr" begin
        x = pytype("Foo", (), ())()
        @test_throws PyException pysetattr(pybuiltins.None, "not_an_attr", 123)
        pysetattr(x, "foo", 123)
        @test pyeq(Bool, pygetattr(x, "foo"), 123)
    end
    @testset "pydelattr" begin
        x = pytype("Foo", (), ())()
        @test !pyhasattr(x, "foo")
        pysetattr(x, "foo", 123)
        @test pyhasattr(x, "foo")
        pydelattr(x, "foo")
        @test !pyhasattr(x, "foo")
        @test_throws PyException pydelattr(pybuiltins.None, "__class__")
        @test_throws PyException pydelattr(pybuiltins.None, "not_an_attr")
    end
    @testset "pyissubclass" begin
        @test pyissubclass(pybuiltins.object, pybuiltins.object)
        @test pyissubclass(pybuiltins.int, pybuiltins.object)
        @test pyissubclass(pybuiltins.bool, pybuiltins.object)
        @test pyissubclass(pybuiltins.bool, pybuiltins.int)
        @test !pyissubclass(pybuiltins.int, pybuiltins.bool)
        @test !pyissubclass(pybuiltins.object, pybuiltins.int)
        @test !pyissubclass(pybuiltins.object, pybuiltins.bool)
        @test_throws PyException pyissubclass(pybuiltins.None, pybuiltins.None)
        @test_throws PyException pyissubclass(pybuiltins.object, pybuiltins.None)
        @test_throws PyException pyissubclass(pybuiltins.None, pybuiltins.object)
    end
    @testset "pyisinstance" begin
        o = pybuiltins.object()
        i = pybuiltins.int()
        b = pybuiltins.bool()
        @test pyisinstance(o, pybuiltins.object)
        @test pyisinstance(i, pybuiltins.object)
        @test pyisinstance(b, pybuiltins.object)
        @test pyisinstance(b, pybuiltins.int)
        @test !pyisinstance(i, pybuiltins.bool)
        @test !pyisinstance(o, pybuiltins.int)
        @test !pyisinstance(o, pybuiltins.bool)
        @test_throws PyException pyisinstance(o, pybuiltins.None)
    end
    @testset "pyhash" begin
        @test pyhash(pybuiltins.None) isa Integer
        @test pyhash(pybuiltins.None) == pyhash(pybuiltins.None)
        @test pyhash(pytuple((1, 2, 3))) == pyhash(pytuple((1, 2, 3)))
    end
    @testset "pytruth" begin
        @test !pytruth(pybuiltins.None)
        @test pytruth(pybuiltins.object)
        for n = -2:2
            @test pytruth(pyint(n)) == (n != 0)
        end
        for n = 0:3
            @test pytruth(pylist(zeros(n))) == (n != 0)
        end
    end
    @testset "pynot" begin
        @test pynot(pybuiltins.None)
        @test !pynot(pybuiltins.int)
        for n = -2:2
            @test pynot(pyfloat(n)) == (n == 0)
        end
        for n = 0:3
            @test pynot(pylist(zeros(n))) == (n == 0)
        end
    end
    @testset "pylen" begin
        for n = 0:3
            @test pylen(pytuple(zeros(n))) == n
            @test pylen(pylist(zeros(n))) == n
        end
        @test_throws PyException pylen(pybuiltins.None)
        @test_throws PyException pylen(pyint(0))
    end
    @testset "pygetitem" begin
        x = pyrange(3)
        @test_throws PyException pygetitem(x, pybuiltins.None)
        @test_throws PyException pygetitem(x, 3)
        @test_throws PyException pygetitem(x, -4)
        @test pyeq(Bool, pygetitem(x, 0), 0)
        @test pyeq(Bool, pygetitem(x, 1), 1)
        @test pyeq(Bool, pygetitem(x, 2), 2)
        @test pyeq(Bool, pygetitem(x, -1), 2)
        @test pyeq(Bool, pygetitem(x, -2), 1)
        @test pyeq(Bool, pygetitem(x, -3), 0)
    end
    @testset "pysetitem" begin
        x = pydict()
        @test_throws PyException pygetitem(x, "foo")
        pysetitem(x, "foo", 123)
        @test pyeq(Bool, pygetitem(x, "foo"), 123)
        pysetitem(x, "foo", 0)
        @test pyeq(Bool, pygetitem(x, "foo"), 0)
    end
    @testset "pydelitem" begin
        x = pydict()
        pysetitem(x, "foo", 123)
        @test pylen(x) == 1
        pydelitem(x, "foo")
        @test pylen(x) == 0
    end
    @testset "pydir" begin
        x = pytype("Foo", (), ["foo" => 1, "bar" => 2])()
        d = pydir(x)
        @test pycontains(d, "__class__")
        @test pycontains(d, "foo")
        @test pycontains(d, "bar")
        @test !pycontains(d, "baz")
    end
    @testset "pycall" begin
        @test pyeq(Bool, pycall(pybuiltins.int), 0)
        @test_throws PyException pycall(pybuiltins.None)
        @test pyeq(Bool, pycall(pybuiltins.int, "10"), 10)
        @test_throws PyException pycall(pybuiltins.int, "10", "20")
        @test pyeq(Bool, pycall(pybuiltins.int, "10", base = 16), 16)
        @test_throws PyException pycall(pybuiltins.int, "10", base = "16")
        @test_throws PyException pycall(pybuiltins.int, "10", bad_argument = 0)
    end
    @testset "pyeq" begin
        @test pyis(pyeq(pybuiltins.None, pybuiltins.None), pybuiltins.True)
        @test pyeq(Bool, pybuiltins.None, pybuiltins.None)
        @test pyis(pyeq(pybuiltins.None, pybuiltins.int), pybuiltins.False)
        @test !pyeq(Bool, pybuiltins.None, pybuiltins.int)
    end
    @testset "pyne" begin
        @test pyis(pyne(pybuiltins.None, pybuiltins.None), pybuiltins.False)
        @test !pyne(Bool, pybuiltins.None, pybuiltins.None)
        @test pyis(pyne(pybuiltins.None, pybuiltins.int), pybuiltins.True)
        @test pyne(Bool, pybuiltins.None, pybuiltins.int)
    end
    @testset "pylt" begin
        for a = -1:1
            for b = -1:1
                @test pyis(pylt(pyint(a), pyint(b)), pybool(a < b))
                @test pylt(Bool, pyint(a), pyint(b)) == (a < b)
            end
        end
    end
    @testset "pyle" begin
        for a = -1:1
            for b = -1:1
                @test pyis(pyle(pyint(a), pyint(b)), pybool(a ≤ b))
                @test pyle(Bool, pyint(a), pyint(b)) == (a ≤ b)
            end
        end
    end
    @testset "pygt" begin
        for a = -1:1
            for b = -1:1
                @test pyis(pygt(pyint(a), pyint(b)), pybool(a > b))
                @test pygt(Bool, pyint(a), pyint(b)) == (a > b)
            end
        end
    end
    @testset "pyge" begin
        for a = -1:1
            for b = -1:1
                @test pyis(pyge(pyint(a), pyint(b)), pybool(a ≥ b))
                @test pyge(Bool, pyint(a), pyint(b)) == (a ≥ b)
            end
        end
    end
    @testset "pycontains" begin
        x = pyrange(3)
        @test pycontains(x, 0)
        @test pycontains(x, 1)
        @test pycontains(x, 2)
        @test !pycontains(x, 3)
        @test !pycontains(x, -1)
        @test !pycontains(x, pybuiltins.None)
    end
    @testset "pyin" begin
        x = pyrange(3)
        @test pyin(0, x)
        @test pyin(1, x)
        @test pyin(2, x)
        @test !pyin(3, x)
        @test !pyin(-1, x)
        @test !pyin(pybuiltins.None, x)
    end
    @testset "getdoc" begin
        @test Base.Docs.getdoc(Py(nothing)) isa Markdown.MD
        @test Base.Docs.getdoc(Py(12)) isa Markdown.MD
        @test Base.Docs.getdoc(pybuiltins.int) isa Markdown.MD
        @test Base.Docs.getdoc(PythonCall.PyNULL) === nothing
    end
end

@testitem "iter" begin
    @test_throws PyException pyiter(pybuiltins.None)
    @test_throws PyException pyiter(pybuiltins.True)
    # unsafe_pynext
    it = pyiter(pyrange(2))
    x = PythonCall.unsafe_pynext(it)
    @test !PythonCall.pyisnull(x)
    @test pyeq(Bool, x, 0)
    x = PythonCall.unsafe_pynext(it)
    @test !PythonCall.pyisnull(x)
    @test pyeq(Bool, x, 1)
    x = PythonCall.unsafe_pynext(it)
    @test PythonCall.pyisnull(x)
    # pynext
    it = pyiter(pyrange(2))
    x = pynext(it)
    @test pyeq(Bool, x, 0)
    x = pynext(it)
    @test pyeq(Bool, x, 1)
    @test_throws PyException pynext(it)
end

@testitem "number" begin
    @testset "pyneg" begin
        for n = -2:2
            @test pyeq(Bool, pyneg(pyint(n)), pyint(-n))
        end
    end
    @testset "pypos" begin
        for n = -2:2
            @test pyeq(Bool, pypos(pyint(n)), pyint(n))
        end
    end
    @testset "pyabs" begin
        for n = -2:2
            @test pyeq(Bool, pyabs(pyint(n)), pyint(abs(n)))
        end
    end
    @testset "pyinv" begin
        for n = -2:2
            @test pyeq(Bool, pyinv(pyint(n)), pyint(-n - 1))
        end
    end
    @testset "pyindex" begin
        for n = -2:2
            @test pyeq(Bool, pyindex(pyint(n)), pyint(n))
        end
    end
    @testset "pyadd" begin
        for x = -2:2
            for y = -2:2
                @test pyeq(Bool, pyadd(pyint(x), pyint(y)), pyint(x + y))
            end
        end
    end
    @testset "pysub" begin
        for x = -2:2
            for y = -2:2
                @test pyeq(Bool, pysub(pyint(x), pyint(y)), pyint(x - y))
            end
        end
    end
    @testset "pymul" begin
        for x = -2:2
            for y = -2:2
                @test pyeq(Bool, pymul(pyint(x), pyint(y)), pyint(x * y))
            end
        end
    end
    # TODO
    # @testset "pymatmul" begin
    #     for x in -2:2
    #         for y in -2:2
    #             @test pyeq(Bool, pymul(pyint(x), pyint(y)), pyint(x*y))
    #         end
    #     end
    # end
    @testset "pytruediv" begin
        for x = -2:2
            for y = -2:2
                if y == 0
                    @test_throws PyException pytruediv(pyint(x), pyint(y))
                else
                    @test pyeq(Bool, pytruediv(pyint(x), pyint(y)), pyfloat(x / y))
                end
            end
        end
    end
    @testset "pyfloordiv" begin
        for x = -2:2
            for y = -2:2
                if y == 0
                    @test_throws PyException pyfloordiv(pyint(x), pyint(y))
                else
                    @test pyeq(Bool, pyfloordiv(pyint(x), pyint(y)), pyfloat(fld(x, y)))
                end
            end
        end
    end
    @testset "pymod" begin
        for x = -2:2
            for y = -2:2
                if y == 0
                    @test_throws PyException pymod(pyint(x), pyint(y))
                else
                    @test pyeq(Bool, pymod(pyint(x), pyint(y)), pyint(mod(x, y)))
                end
            end
        end
    end
    @testset "pydivmod" begin
        for x = -2:2
            for y = -2:2
                if y == 0
                    @test_throws PyException pydivmod(pyint(x), pyint(y))
                else
                    @test pyeq(Bool, pydivmod(pyint(x), pyint(y)), pytuple(fldmod(x, y)))
                end
            end
        end
    end
    @testset "pylshift" begin
        for n = 0:3
            @test pyeq(Bool, pylshift(pyint(123), pyint(n)), pyint(123 << n))
        end
    end
    @testset "pyrshift" begin
        for n = 0:3
            @test pyeq(Bool, pyrshift(pyint(123), pyint(n)), pyint(123 >> n))
        end
    end
    @testset "pyand" begin
        for x = 0:3
            for y = 0:3
                @test pyeq(Bool, pyand(pyint(x), pyint(y)), pyint(x & y))
            end
        end
    end
    @testset "pyxor" begin
        for x = 0:3
            for y = 0:3
                @test pyeq(Bool, pyxor(pyint(x), pyint(y)), pyint(x ⊻ y))
            end
        end
    end
    @testset "pyor" begin
        for x = 0:3
            for y = 0:3
                @test pyeq(Bool, pyor(pyint(x), pyint(y)), pyint(x | y))
            end
        end
    end
    # TODO: in-place operators
end

@testitem "builtins" begin
    @testset "pyprint" begin
        buf = pyimport("io").StringIO()
        ans = pyprint("hello", 12, file = buf)
        @test ans === nothing
        buf.seek(0)
        @test pyeq(Bool, buf.read().strip(), "hello 12")
    end
    @testset "pyall" begin
        for val in [[true, true], [true, false], [false, false]]
            @test pyall(pylist(val)) === all(val)
        end
    end
    @testset "pyany" begin
        for val in [[true, true], [true, false], [false, false]]
            @test pyany(pylist(val)) === any(val)
        end
    end
    @testset "pycallable" begin
        @test pycallable(pybuiltins.str) === true
        @test pycallable(pybuiltins.any) === true
        @test pycallable(pybuiltins.None) === false
        @test pycallable(pybuiltins.True) === false
        @test pycallable(pybuiltins.object) === true
    end
    @testset "pycompile" begin
        ans = pycompile("3+4", "foo.py", "eval")
        @test pyeq(Bool, ans.co_filename, "foo.py")
        @test pyeq(Bool, pybuiltins.eval(ans, pydict()), 7)
    end
end

@testitem "import" begin
    sys = pyimport("sys")
    os = pyimport("os")
    @test pyeq(Bool, sys.__name__, "sys")
    @test pyeq(Bool, os.__name__, "os")
    sysos = pyimport("sys", "os")
    @test sysos isa Tuple{Py,Py}
    @test pyis(sysos[1], sys)
    @test pyis(sysos[2], os)
    ver = pyimport("sys" => "version")
    @test pyis(ver, sys.version)
    path = pyimport("sys" => "path")
    @test pyis(path, sys.path)
    verpath = pyimport("sys" => ("version", "path"))
    @test verpath isa Tuple{Py,Py}
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
    @test pyisinstance(pybytes(UInt8[1, 2, 3]), pybuiltins.bytes)
    @test pyeq(Bool, pybytes(pylist([1, 2, 3])), pybytes(UInt8[1, 2, 3]))
    @test pyeq(Bool, pybytes(b"foo"), pystr("foo").encode("ascii"))
    @test pyeq(
        Bool,
        pybytes(codeunits(SubString("foobarbaz", 4:6))),
        pystr("bar").encode("ascii"),
    )
    @test pybytes(Vector, pylist([1, 2, 3])) == UInt8[1, 2, 3]
    @test pybytes(Vector{UInt8}, pylist([1, 2, 3])) == UInt8[1, 2, 3]
    @test pybytes(Base.CodeUnits, pystr("foo").encode("ascii")) == b"foo"
    @test pybytes(Base.CodeUnits{UInt8,String}, pystr("bar").encode("ascii")) == b"bar"
end

@testitem "tuple" begin
    z = pytuple()
    @test pyisinstance(z, pybuiltins.tuple)
    @test pylen(z) == 0
    x = pytuple((1, 2, 3))
    @test pyisinstance(x, pybuiltins.tuple)
    @test pylen(x) == 3
    @test pyeq(Bool, pygetitem(x, 0), 1)
    @test pyeq(Bool, pygetitem(x, 1), 2)
    @test pyeq(Bool, pygetitem(x, 2), 3)
    @test pyeq(Bool, pytuple([1, 2, 3]), x)
    @test pyeq(Bool, pytuple(i + 1 for i = 0:10 if i < 3), x)
    @test pyeq(Bool, pytuple(pytuple((1, 2, 3))), x)
    @test pyeq(Bool, pytuple(pylist([1, 2, 3])), x)
end

@testitem "list" begin
    z = pylist()
    @test pyisinstance(z, pybuiltins.list)
    @test pylen(z) == 0
    x = pylist((1, 2, 3))
    @test pyisinstance(x, pybuiltins.list)
    @test pylen(x) == 3
    @test pyeq(Bool, pygetitem(x, 0), 1)
    @test pyeq(Bool, pygetitem(x, 1), 2)
    @test pyeq(Bool, pygetitem(x, 2), 3)
    @test pyeq(Bool, pylist([1, 2, 3]), x)
    @test pyeq(Bool, pylist(i + 1 for i = 0:10 if i < 3), x)
    @test pyeq(Bool, pylist(pylist((1, 2, 3))), x)
    @test pyeq(Bool, pylist(pytuple([1, 2, 3])), x)
    @test pyeq(Bool, pycollist([1, 2, 3]), pylist([1, 2, 3]))
    @test pyeq(Bool, pycollist([1 2; 3 4]), pylist((pylist([1, 3]), pylist([2, 4]))))
    @test pyeq(Bool, pyrowlist([1, 2, 3]), pylist([1, 2, 3]))
    @test pyeq(Bool, pyrowlist([1 2; 3 4]), pylist((pylist([1, 2]), pylist([3, 4]))))
end

@testitem "dict" begin
    z = pydict()
    @test pyisinstance(z, pybuiltins.dict)
    @test pylen(z) == 0
    x = pydict(foo = 1, bar = 2)
    @test pyisinstance(x, pybuiltins.dict)
    @test pylen(x) == 2
    @test pyeq(Bool, pygetitem(x, "foo"), 1)
    @test pyeq(Bool, pygetitem(x, "bar"), 2)
    @test pyeq(Bool, pydict(["foo" => 1, "bar" => 2]), x)
    @test pyeq(Bool, pydict([("foo" => 1), ("bar" => 2)]), x)
    @test pyeq(Bool, pydict(Dict("foo" => 1, "bar" => 2)), x)
    @test pyeq(Bool, pydict((foo = 1, bar = 2)), x)
    @test pyeq(Bool, pydict(x), x)
    @test pyeq(Bool, pydict("foo" => 1, "bar" => 2), x)
end

@testitem "bool" begin
    @test pyis(pybool(), pybuiltins.False)
    @test pyis(pybool(false), pybuiltins.False)
    @test pyis(pybool(true), pybuiltins.True)
    @test pyis(pybool(0.0), pybuiltins.False)
    @test pyis(pybool(-1.2), pybuiltins.True)
    @test pyis(pybool(pybuiltins.None), pybuiltins.False)
    @test pyis(pybool(pylist()), pybuiltins.False)
    @test pyis(pybool(pylist([1, 2, 3])), pybuiltins.True)
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
    x = 1 // 4
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
    x = [1, 2, 3, 2, 1]
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
    x = pytype("Foo", (), ["foo" => 1, "bar" => 2])
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
    @test pyeq(Bool, pyeval("x+1", g, (x = 1,)), 2)
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
    # @pyexec
    @test_throws PyException @pyexec(`raise ValueError`)
    @test @pyexec(`1 + 2`) === nothing
    @test @pyexec(`ans = 1 + 2` => (ans::Int,)) === (ans = 3,)
    @test @pyexec((x = 1, y = 2) => `ans = x + y` => (ans::Int,)) === (ans = 3,)
    # @pyeval
    @test_throws PyException @pyeval(`import sys`)  # not an expression
    @test_throws PyException @pyeval(`None + None`)
    @test @pyeval(`1 + 2` => Int) === 3
    @test @pyeval((x = 1, y = 2) => `x + y` => Int) === 3
end

@testitem "@pyconst" begin
    f() = @pyconst "hello"
    g() = @pyconst "hello"
    @test f() === f()
    @test f() === f()
    @test g() === g()
    @test g() !== f()
    @test f() isa Py
    @test pyeq(Bool, f(), "hello")
end

@testitem "Base.jl" begin
    @testset "broadcast" begin
        # Py always broadcasts as a scalar
        x = [1 2; 3 4] .+ Py(1)
        @test isequal(x, [Py(2) Py(3); Py(4) Py(5)])
        x = Py("foo") .* [1 2; 3 4]
        @test isequal(x, [Py("foo") Py("foofoo"); Py("foofoofoo") Py("foofoofoofoo")])
        # this previously treated the list as a shape (2,) object
        # but now tries to do `1 + [1, 2]` which properly fails
        @test_throws PyException [1 2; 3 4] .+ pylist([1, 2])
    end
    @testset "showable" begin
        @test showable(MIME("text/plain"), Py(nothing))
        @test showable(MIME("text/plain"), Py(12))
        # https://github.com/JuliaPy/PythonCall.jl/issues/522
        @test showable(MIME("text/plain"), PythonCall.pynew())
        @test !showable(MIME("text/html"), PythonCall.pynew())
    end
    @testset "show" begin
        @test sprint(show, MIME("text/plain"), Py(nothing)) == "Python: None"
        @test sprint(show, MIME("text/plain"), Py(12)) == "Python: 12"
        # https://github.com/JuliaPy/PythonCall.jl/issues/522
        @test sprint(show, MIME("text/plain"), PythonCall.pynew()) == "Python: NULL"
        @test_throws MethodError sprint(show, MIME("text/html"), PythonCall.pynew())
    end
end

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

@testitem "propertynames" begin
    x = pyint(7)
    task = Threads.@spawn propertynames(x)
    properties = propertynames(x)
    @test :__init__ in properties
    prop_task = fetch(task)
    @test properties == prop_task
end
