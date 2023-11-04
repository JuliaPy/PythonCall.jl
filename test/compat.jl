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

@testitem "gui" begin
    @testset "fix_qt_plugin_path" begin
        @test PythonCall.fix_qt_plugin_path() isa Bool
        # second time is a no-op
        @test PythonCall.fix_qt_plugin_path() === false
    end
    @testset "event_loop_on/off" begin
        for g in [:pyqt4, :pyqt5, :pyside, :pyside2, :gtk, :gtk3, :wx]
            # TODO: actually test the various GUIs somehow?
            @show g
            @test_throws PyException PythonCall.event_loop_on(g)
            @test PythonCall.event_loop_off(g) === nothing
        end
    end
end

@testitem "ipython" begin
    @testset "PythonDisplay" begin
        sys = pyimport("sys")
        io = pyimport("io")
        pystdout = sys.stdout
        fp = sys.stdout = io.StringIO()
        try
            d = PythonCall._compat.PythonDisplay()
            @test display(d, 123) === nothing
            fp.seek(0)
            @test pyconvert(String, fp.read()) == "123\n"
        finally
            sys.stdout = pystdout
        end
    end
    @testset "IPythonDisplay" begin
        # TODO
    end
end

@testitem "multimedia" begin
    # TODO
end

@testitem "PyCall.jl" begin
    # TODO
end

@testitem "Serialization.jl" begin
    using Serialization
    @testset "Py" begin
        for x in Py[Py(123), Py(1.23), Py("hello"), pylist([1, 2, 3]), pytuple([1, 2, 3]), Py(nothing), Py([1, 2, 3]), Py(:hello)]
            io = IOBuffer()
            serialize(io, x)
            seekstart(io)
            y = deserialize(io)
            @test y isa Py
            @test pyis(pytype(x), pytype(y))
            @test pyeq(Bool, x, y)
        end            
    end
    @testset "PyException" begin
        for e in Py[pybuiltins.ValueError("hello")]
            io = IOBuffer()
            x = PyException(e)
            serialize(io, x)
            seekstart(io)
            y = deserialize(io)
            @test y isa PyException
            @test pyeq(Bool, y.t, x.t)
            @test pyeq(Bool, y.v.args, x.v.args)
        end
    end
end

@testitem "Tables.jl" begin
    @testset "pytable" begin
        x = (x = [1,2,3], y = ["a", "b", "c"])
        # pandas
        # TODO: install pandas and test properly
        @test_throws PyException pytable(x, :pandas)
        # columns
        y = pytable(x, :columns)
        @test pyeq(Bool, y, pydict(x=[1,2,3], y=["a","b","c"]))
        # rows
        y = pytable(x, :rows)
        @test pyeq(Bool, y, pylist([(1, "a"), (2, "b"), (3, "c")]))
        @test all(pyisinstance(y.x, pybuiltins.int) for y in y)
        @test all(pyisinstance(y.y, pybuiltins.str) for y in y)
        # rowdicts
        y = pytable(x, :rowdicts)
        @test pyeq(Bool, y, pylist([pydict(x=1, y="a"), pydict(x=2, y="b"), pydict(x=3, y="c")]))
    end
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

@testitem "@py" begin
    @testset "literals" begin
        # int
        x = @py(123)
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.int)
        @test pyeq(Bool, x, 123)
        # uint
        x = @py(0x123)
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.int)
        @test pyeq(Bool, x, 0x123)
        # TODO: these don't work on all platforms??
        # # int128
        # x = @py(12345678901234567890)
        # @test x isa Py
        # @test pyis(pytype(x), pybuiltins.int)
        # @test pyeq(Bool, x, 12345678901234567890)
        # # uint128
        # x = @py(0x12345678901234567890)
        # @test x isa Py
        # @test pyis(pytype(x), pybuiltins.int)
        # @test pyeq(Bool, x, 0x12345678901234567890)
        # # bigint
        # x = @py(big"1234567890123456789012345678901234567890")
        # @test x isa Py
        # @test pyis(pytype(x), pybuiltins.int)
        # @test pyeq(Bool, x, big"1234567890123456789012345678901234567890")
        # x = @py(1234567890123456789012345678901234567890)
        # @test x isa Py
        # @test pyis(pytype(x), pybuiltins.int)
        # @test pyeq(Bool, x, big"1234567890123456789012345678901234567890")
        # None
        x = @py(None)
        @test pyis(x, pybuiltins.None)
        # True
        x = @py(True)
        @test x isa Py
        @test pyis(x, pybuiltins.True)
        x = @py(true)
        @test x isa Py
        @test pyis(x, pybuiltins.True)
        # False
        x = @py(False)
        @test x isa Py
        @test pyis(x, pybuiltins.False)
        x = @py(false)
        @test x isa Py
        @test pyis(x, pybuiltins.False)
        # str
        x = @py("hello")
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.str)
        @test pyeq(Bool, x, "hello")
        # float
        x = @py(1.23)
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.float)
        @test pyeq(Bool, x, 1.23)
        # tuple
        x = @py tuple()
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.tuple)
        @test pyeq(Bool, x, ())
        x = @py (1, 2, 3)
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.tuple)
        @test pyeq(Bool, x, (1, 2, 3))
        # list
        x = @py list()
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.list)
        @test pyeq(Bool, x, pylist())
        x = @py [1, 2, 3]
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.list)
        @test pyeq(Bool, x, pylist([1, 2, 3]))
        # dict
        x = @py dict()
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict())
        x = @py {}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict())
        x = @py {x=1, y=2}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict(x=1, y=2))
        x = @py {"x": 1, "y": 2}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict(x=1, y=2))
        # set
        x = @py set()
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.set)
        @test pyeq(Bool, x, pyset())
        x = @py {1, 2, 3}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.set)
        @test pyeq(Bool, x, pyset([1, 2, 3]))
    end
    @testset "__file__" begin
        x = @py __file__
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.str)
        @test pyeq(Bool, x, @__FILE__)
    end
    @testset "__line__" begin
        x = @py __line__
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.int)
        @test pyeq(Bool, x, @__LINE__() - 3)
    end
    @testset "builtins" begin
        x = @py int
        @test pyis(x, pybuiltins.int)
        x = @py float
        @test pyis(x, pybuiltins.float)
        x = @py ValueError
        @test pyis(x, pybuiltins.ValueError)
    end
    @testset "variables" begin
        x = 123
        y = @py x
        @test y === x
    end
    @testset "arithmetic" begin
        x = @py 1 + 2 + 3
        @test pyeq(Bool, x, 6)
        x = @py "foo" + "bar"
        @test pyeq(Bool, x, "foobar")
        x = @py 9 - 2
        @test pyeq(Bool, x, 7)
        x = @py 2 * 3 * 4
        @test pyeq(Bool, x, 24)
        x = @py "foo" * 3
        @test pyeq(Bool, x, "foofoofoo")
        x = @py 10 / 4
        @test pyeq(Bool, x, 2.5)
        x = @py 1 << 10
        @test pyeq(Bool, x, 1024)
        x = @py -(10)
        @test pyeq(Bool, x, -10)
    end
    @testset "attrs" begin
        t = pytype("Test", (pybuiltins.object,), [])
        x = t()
        @py x.foo = "foo"
        @test pyeq(Bool, x.foo, "foo")
    end
    @testset "items" begin
        x = pylist([1, 2, 3])
        @test pyeq(Bool, @py(x[0]), 1)
        @test pyeq(Bool, @py(x[-1]), 3)
        @test pyeq(Bool, @py(x[0:2]), pylist([1, 2]))
        @py x[1] = 0
        @test pyeq(Bool, x, pylist([1, 0, 3]))
    end
    @testset "assign" begin
        @py x = 12
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.int)
        @test pyeq(Bool, x, 12)
    end
    @testset "@jl" begin
        x = @py @jl "foo"^3
        @test x == "foofoofoo"
    end
    @testset "begin" begin
        z = @py begin
            x = "foo"
            y = 4
            x * y
        end
        @test x isa Py
        @test pyeq(Bool, x, "foo")
        @test y isa Py
        @test pyeq(Bool, y, 4)
        @test z isa Py
        @test pyeq(Bool, z, "foofoofoofoo")
    end
    @testset "import" begin
        @py import sys
        @test pyis(sys, pyimport("sys"))
        @py import sys as _sys
        @test pyis(_sys, sys)
        @py import sys as _sys2, sys as _sys3
        @test pyis(_sys2, sys)
        @test pyis(_sys3, sys)
        @py import sys: version_info
        @test pyis(version_info, sys.version_info)
        @py import sys: modules as _mods, version_info as _ver
        @test pyis(_mods, sys.modules)
        @test pyis(_ver, sys.version_info)
    end
    @testset "short-circuit" begin
        x = @py 3 && pylist([1,2])
        @test pyeq(Bool, x, pylist([1, 2]))
        x = @py None && True
        @test pyis(x, pybuiltins.None)
        x = @py None || 0 || pyset()
        @test pyeq(Bool, x, pyset())
        x = @py pydict() || 8 || ""
        @test pyeq(Bool, x, 8)
    end
    @testset "if" begin
        x = @py if 1 == 2; "a"; end
        @test x isa Py
        @test pyis(x, pybuiltins.None)
        x = @py if 1 < 2; "a"; end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2; "a"; else; "b"; end
        @test x isa Py
        @test pyeq(Bool, x, "b")
        x = @py if 1 < 2; "a"; else; "b"; end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2; "a"; elseif 1 < 2; "b"; end
        @test x isa Py
        @test pyeq(Bool, x, "b")
        x = @py if 1 < 2; "a"; elseif 2 < 3; "b"; end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2; "a"; elseif 2 == 3; "b"; end
        @test x isa Py
        @test pyis(x, pybuiltins.None)
    end
    @testset "for" begin
        x = pydict(x=1, y=2)
        y = pylist()
        @py for k in x
            y.append(k)
        end
        @test pyeq(Bool, y, pylist(["x", "y"]))
    end
    @testset "while" begin
        x = pylist([1, 2, 3, 4])
        y = pylist()
        @py while len(x) > 2
            y.append(x.pop())
        end
        @test pyeq(Bool, x, pylist([1, 2]))
        @test pyeq(Bool, y, pylist([4, 3]))
    end
    @testset "string interpolation" begin
        x = @py """$(None)$(True)$("foo"*2)"""
        @test pyeq(Bool, x, "NoneTruefoofoo")
    end
end
