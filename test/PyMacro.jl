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
        x = @py {x = 1, y = 2}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict(x = 1, y = 2))
        x = @py {"x":1, "y":2}
        @test x isa Py
        @test pyis(pytype(x), pybuiltins.dict)
        @test pyeq(Bool, x, pydict(x = 1, y = 2))
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
        x = @py 3 && @jl(pylist([1, 2]))
        @test pyeq(Bool, x, pylist([1, 2]))
        x = @py None && True
        @test pyis(x, pybuiltins.None)
        x = @py None || 0 || @jl(pyset())
        @test pyeq(Bool, x, pyset())
        x = @py @jl(pydict()) || 8 || ""
        @test pyeq(Bool, x, 8)
    end
    @testset "if" begin
        x = @py if 1 == 2
            "a"
        end
        @test x isa Py
        @test pyis(x, pybuiltins.None)
        x = @py if 1 < 2
            "a"
        end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2
            "a"
        else
            "b"
        end
        @test x isa Py
        @test pyeq(Bool, x, "b")
        x = @py if 1 < 2
            "a"
        else
            "b"
        end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2
            "a"
        elseif 1 < 2
            "b"
        end
        @test x isa Py
        @test pyeq(Bool, x, "b")
        x = @py if 1 < 2
            "a"
        elseif 2 < 3
            "b"
        end
        @test x isa Py
        @test pyeq(Bool, x, "a")
        x = @py if 1 == 2
            "a"
        elseif 2 == 3
            "b"
        end
        @test x isa Py
        @test pyis(x, pybuiltins.None)
    end
    @testset "for" begin
        x = pydict(x = 1, y = 2)
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
