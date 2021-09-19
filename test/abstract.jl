@testset "object" begin
    @testset "pyis" begin
        x = pylist()
        y = Py(x)
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
        @test pyisinstance(pygetattr(pybuiltins.None, "__class__", pybuiltins.None), pybuiltins.type)
        @test pyis(pygetattr(pybuiltins.None, "not_an_attr", pybuiltins.None), pybuiltins.None)
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
        for n in -2:2
            @test pytruth(pyint(n)) == (n != 0)
        end
        for n in 0:3
            @test pytruth(pylist(zeros(n))) == (n != 0)
        end
    end
    @testset "pynot" begin
        @test pynot(pybuiltins.None)
        @test !pynot(pybuiltins.int)
        for n in -2:2
            @test pynot(pyfloat(n)) == (n == 0)
        end
        for n in 0:3
            @test pynot(pylist(zeros(n))) == (n == 0)
        end
    end
    @testset "pylen" begin
        for n in 0:3
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
        x = pytype("Foo", (), (foo=1, bar=2))()
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
        @test pyeq(Bool, pycall(pybuiltins.int, "10", base=16), 16)
        @test_throws PyException pycall(pybuiltins.int, "10", base="16")
        @test_throws PyException pycall(pybuiltins.int, "10", bad_argument=0)
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
        for a in -1:1
            for b in -1:1
                @test pyis(pylt(pyint(a), pyint(b)), pybool(a < b))
                @test pylt(Bool, pyint(a), pyint(b)) == (a < b)
            end
        end
    end
    @testset "pyle" begin
        for a in -1:1
            for b in -1:1
                @test pyis(pyle(pyint(a), pyint(b)), pybool(a ≤ b))
                @test pyle(Bool, pyint(a), pyint(b)) == (a ≤ b)
            end
        end
    end
    @testset "pygt" begin
        for a in -1:1
            for b in -1:1
                @test pyis(pygt(pyint(a), pyint(b)), pybool(a > b))
                @test pygt(Bool, pyint(a), pyint(b)) == (a > b)
            end
        end
    end
    @testset "pyge" begin
        for a in -1:1
            for b in -1:1
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
end
