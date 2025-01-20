@testitem "any" begin
    mutable struct Foo
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
    Base.powermod(x::Foo, y::Foo, z::Foo) = "powermod($(x.value), $(y.value), $(z.value))"
    (x::Foo)(args...; kw...) = "$(x.value)($args)$(length(kw))"
    Base.getindex(x::Foo, idx...) = "$(x.value)[$idx]"
    Base.setindex!(x::Foo, v, idx...) = (x.value = v + sum(idx); x)
    Base.delete!(x::Foo, idx...) = (x.value = -sum(idx); x)
    Base.in(v::Int, x::Foo) = x.value == v
    Base.nameof(x::Foo) = "nameof $(x.value)"
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
    @testset "repr" begin
        @test pyrepr(String, pyjl(missing)) == "Julia: missing"
    end
    @testset "str" begin
        @test pystr(String, pyjl(missing)) == "missing"
    end
    @testset "getattr" begin
        @test pyconvert(Int, pygetattr(pyjl(Foo(12)), "value")) === 12
    end
    @testset "setattr" begin
        x = Foo(12)
        @test x.value == 12
        pysetattr(x, "value", 34)
        @test x.value == 34
    end
    @testset "dir" begin
        @test pycontains(pydir(pyjl(Foo(99))), "value")
    end
    @testset "call" begin
        z = pyjl(Foo(1))(4, 5)
        @test pyconvert(String, z) == "1((4, 5))0"
        z = pyjl(Foo(1))(4, 5; foo = true, bar = true)
        @test pyconvert(String, z) == "1((4, 5))2"
    end
    @testset "getitem" begin
        z = pygetitem(pyjl(Foo(1)), 3)
        @test pyconvert(String, z) == "1[(3,)]"
        z = pygetitem(pyjl(Foo(1)), (4, 5))
        @test pyconvert(String, z) == "1[(4, 5)]"
    end
    @testset "setitem" begin
        z = Foo(0)
        x = pyjl(z)
        pysetitem(x, 10, 3)
        @test z.value == 13
        pysetitem(x, (10, 10), 4)
        @test z.value == 24
    end
    @testset "delitem" begin
        z = Foo(0)
        x = pyjl(z)
        pydelitem(x, 9)
        @test z.value == -9
        pydelitem(x, (3, 4))
        @test z.value == -7
    end
    @testset "contains" begin
        @test pycontains(pyjl(Foo(45)), 45)
    end
    @testset "pos" begin
        z = +(pyjl(Foo(1)))
        @test pyconvert(String, z) == "+ 1"
    end
    @testset "neg" begin
        z = -(pyjl(Foo(1)))
        @test pyconvert(String, z) == "- 1"
    end
    @testset "abs" begin
        z = abs(pyjl(Foo(1)))
        @test pyconvert(String, z) == "abs 1"
    end
    @testset "inv" begin
        z = ~(pyjl(Foo(1)))
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
        z = pyjl(Foo(1))^pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ^ 2"
    end
    @testset "rpow" begin
        z = pyjlraw(Foo(1))^pyjl(Foo(2))
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
    @testset "pow3" begin
        z = pypow(pyjl(Foo(1)), pyjl(Foo(2)), pyjl(Foo(3)))
        @test pyconvert(String, z) == "powermod(1, 2, 3)"
    end
    @testset "rpow3" begin
        z = pyjl(Foo(2)).__rpow__(pyjl(Foo(1)), pyjl(Foo(3)))
        @test pyconvert(String, z) == "powermod(1, 2, 3)"
    end
    @testset "name" begin
        z = pyjl(Foo(135)).__name__
        @test pyconvert(String, z) == "nameof 135"
    end
    @testset "mimebundle" begin
        z = pyjl(Foo(1))._repr_mimebundle_()
        @test pyisinstance(z, pybuiltins.dict)
        @test pycontains(z, "text/plain")
    end
    @testset "display" begin
        pyjl(Foo(1))._jl_display()
        pyjl(Foo(1))._jl_display(mime = "text/plain")
    end
    @testset "help" begin
        @test pyis(pyjl(Foo(1))._jl_help(), nothing)
        @test pyis(pyjl(Foo(1))._jl_help(mime = "text/plain"), nothing)
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
    @testset "ndim" begin
        @test pyeq(Bool, pyjl(fill(nothing)).ndim, 0)
        @test pyeq(Bool, pyjl(fill(nothing, 1)).ndim, 1)
        @test pyeq(Bool, pyjl(fill(nothing, 1, 1)).ndim, 2)
        @test pyeq(Bool, pyjl(fill(nothing, 1, 1, 1)).ndim, 3)
    end
    @testset "shape" begin
        @test pyeq(Bool, pyjl(fill(nothing)).shape, ())
        @test pyeq(Bool, pyjl(fill(nothing, 3)).shape, (3,))
        @test pyeq(Bool, pyjl(fill(nothing, 3, 5)).shape, (3, 5))
        @test pyeq(Bool, pyjl(fill(nothing, 3, 5, 2)).shape, (3, 5, 2))
    end
    @testset "getitem" begin
        x = pyjl([1, 2, 3, 4, 5])
        @test pyeq(Bool, x[0], 1)
        @test pyeq(Bool, x[1], 2)
        @test pyeq(Bool, x[2], 3)
        @test pyeq(Bool, x[-1], 5)
        @test pyeq(Bool, x[-2], 4)
        @test pyeq(Bool, x[-3], 3)
        @test pyjlvalue(x[pyslice(3)]) == [1, 2, 3]
        @test pyjlvalue(x[pyslice(2)]) == [1, 2]
        @test pyjlvalue(x[pyslice(1, 2)]) == [2]
        @test pyjlvalue(x[pyslice(2, 2)]) == []
        @test pyjlvalue(x[pyslice(0, -1)]) == [1, 2, 3, 4]
        @test pyjlvalue(x[pyslice(-2, nothing)]) == [4, 5]
        @test pyjlvalue(x[pyslice(0, 3, 1)]) == [1, 2, 3]
        @test pyjlvalue(x[pyslice(nothing, nothing, 2)]) == [1, 3, 5]
        @test pyjlvalue(x[pyslice(1, nothing, 2)]) == [2, 4]
        @test pyjlvalue(x[pyslice(0, nothing, 3)]) == [1, 4]
        x = pyjl([1 2; 3 4])
        @test pyeq(Bool, x[0, 0], 1)
        @test pyeq(Bool, x[0, 1], 2)
        @test pyeq(Bool, x[1, 0], 3)
        @test pyeq(Bool, x[1, 1], 4)
        @test pyjlvalue(x[1, pyslice(nothing)]) == [3, 4]
        @test pyjlvalue(x[pyslice(nothing), 1]) == [2, 4]
    end
    @testset "setitem" begin
        x = [0 0; 0 0]
        y = pyjl(x)
        y[0, 0] = 1
        @test x == [1 0; 0 0]
        y[0, 1] = 2
        @test x == [1 2; 0 0]
        y[1, 0] = 3
        @test x == [1 2; 3 0]
        y[-1, 0] = 4
        @test x == [1 2; 4 0]
        y[-2, pyslice(nothing)] = 5
        @test x == [5 5; 4 0]
        y[pyslice(nothing), -1] = 6
        @test x == [5 6; 4 6]
        y[pyslice(nothing), pyslice(nothing)] = 7
        @test x == [7 7; 7 7]
    end
    @testset "delitem" begin
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = pyjl(x)
        pydelitem(y, 0)
        @test x == [2, 3, 4, 5, 6, 7, 8]
        pydelitem(y, 2)
        @test x == [2, 3, 5, 6, 7, 8]
        pydelitem(y, -3)
        @test x == [2, 3, 5, 7, 8]
        pydelitem(y, pyslice(1, nothing, 2))
        @test x == [2, 5, 8]
    end
    @testset "reshape" begin
        x = pyjl([1, 2, 3, 4, 5, 6, 7, 8])
        @test pyeq(Bool, x.shape, (8,))
        y = x.reshape((2, 4))
        @test pyeq(Bool, y.shape, (2, 4))
        @test pyjlvalue(y) == [1 3 5 7; 2 4 6 8]
    end
    @testset "copy" begin
        x = pyjl([1 2; 3 4])
        y = x.copy()
        @test pyis(pytype(y), PythonCall.pyjlarraytype)
        @test pyjlvalue(x) == pyjlvalue(y)
        @test typeof(pyjlvalue(x)) == typeof(pyjlvalue(y))
        @test pyjlvalue(x) !== pyjlvalue(y)
        x[0, 0] = 0
        @test pyjlvalue(x) == [0 2; 3 4]
        @test pyjlvalue(y) == [1 2; 3 4]
    end
    @testset "array_interface" begin
        x = pyjl(Float32[1 2 3; 4 5 6]).__array_interface__
        @test pyisinstance(x, pybuiltins.dict)
        @test pyeq(Bool, x["shape"], (2, 3))
        @test pyeq(Bool, x["typestr"], "<f4")
        @test pyisinstance(x["data"], pybuiltins.tuple)
        @test pylen(x["data"]) == 2
        @test pyeq(Bool, x["strides"], (4, 8))
        @test pyeq(Bool, x["version"], 3)
    end
    @testset "array_struct" begin
        # TODO (not implemented)
        # x = pyjl(Float32[1 2 3; 4 5 6]).__array_struct__
    end
    @testset "buffer" begin
        m = pybuiltins.memoryview(pyjl(Float32[1 2 3; 4 5 6]))
        @test !pytruth(m.c_contiguous)
        @test pytruth(m.contiguous)
        @test pytruth(m.f_contiguous)
        @test pyeq(Bool, m.format, "f")
        @test pyeq(Bool, m.itemsize, 4)
        @test pyeq(Bool, m.nbytes, 4 * 6)
        @test pyeq(Bool, m.ndim, 2)
        @test !pytruth(m.readonly)
        @test pyeq(Bool, m.shape, (2, 3))
        @test pyeq(Bool, m.strides, (4, 8))
        @test pyeq(Bool, m.suboffsets, ())
        @test pyeq(Bool, m.tolist(), pylist([pylist([1, 2, 3]), pylist([4, 5, 6])]))
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
        @test pytruth(pyjl(Dict("one" => 1, "two" => 2)))
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
    x1 = [1, 2, 3, 4, 5]
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
        @test pyis(pytype(pyjl(0 // 1)), PythonCall.pyjlrationaltype)
        @test pyis(pytype(pyjl(0.0)), PythonCall.pyjlrealtype)
        @test pyis(pytype(pyjl(Complex(0.0))), PythonCall.pyjlcomplextype)
    end
    @testset "bool" begin
        @test !pytruth(pyjl(false))
        @test !pytruth(pyjl(0))
        @test !pytruth(pyjl(0 // 1))
        @test !pytruth(pyjl(0.0))
        @test !pytruth(pyjl(Complex(0.0)))
        @test pytruth(pyjl(true))
        @test pytruth(pyjl(3))
        @test pytruth(pyjl(5 // 2))
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
        @test pytruth(pyjl(Set([1, 2, 3])))
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
        @test pytruth(pyjl([1, 2]))
    end
    @testset "resize" begin
        x = pyjl([1, 2, 3, 4, 5])
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.resize(5)
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.resize(3)
        @test pyjlvalue(x) == [1, 2, 3]
        x.resize(0)
        @test pyjlvalue(x) == []
        x.resize(2)
        x[0] = 5
        x[1] = 6
        @test pyjlvalue(x) == [5, 6]
    end
    @testset "sort" begin
        x = pyjl([4, 6, 2, 3, 7, 6, 1])
        x.sort()
        @test pyjlvalue(x) == [1, 2, 3, 4, 6, 6, 7]
        x = pyjl([4, 6, 2, 3, 7, 6, 1])
        x.sort(reverse = true)
        @test pyjlvalue(x) == [7, 6, 6, 4, 3, 2, 1]
        x = pyjl([4, -6, 2, -3, 7, -6, 1])
        x.sort(key = abs)
        @test pyjlvalue(x) == [1, 2, -3, 4, -6, -6, 7]
        x = pyjl([4, -6, 2, -3, 7, -6, 1])
        x.sort(key = abs, reverse = true)
        @test pyjlvalue(x) == [7, -6, -6, 4, -3, 2, 1]
    end
    @testset "reverse" begin
        x = pyjl([1, 2, 3, 4, 5])
        x.reverse()
        @test pyjlvalue(x) == [5, 4, 3, 2, 1]
    end
    @testset "clear" begin
        x = pyjl([1, 2, 3, 4, 5])
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.clear()
        @test pyjlvalue(x) == []
    end
    @testset "reversed" begin
        x = pyjl([1, 2, 3, 4, 5])
        y = pybuiltins.reversed(x)
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        @test pyjlvalue(y) == [5, 4, 3, 2, 1]
    end
    @testset "insert" begin
        x = pyjl([1, 2, 3])
        x.insert(0, 4)
        @test pyjlvalue(x) == [4, 1, 2, 3]
        x.insert(2, 5)
        @test pyjlvalue(x) == [4, 1, 5, 2, 3]
        x.insert(5, 6)
        @test pyjlvalue(x) == [4, 1, 5, 2, 3, 6]
        x.insert(-3, 7)
        @test pyjlvalue(x) == [4, 1, 5, 7, 2, 3, 6]
        @test_throws PyException x.insert(10, 10)
    end
    @testset "append" begin
        x = pyjl([1, 2, 3])
        x.append(4)
        @test pyjlvalue(x) == [1, 2, 3, 4]
        x.append(5.0)
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        @test_throws PyException x.append(nothing)
        @test_throws PyException x.append(1.2)
        @test_throws PyException x.append("2")
    end
    @testset "extend" begin
        x = pyjl([1, 2, 3])
        x.extend(pylist())
        @test pyjlvalue(x) == [1, 2, 3]
        x.extend(pylist([4, 5]))
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.extend(pylist([6.0]))
        @test pyjlvalue(x) == [1, 2, 3, 4, 5, 6]
    end
    @testset "pop" begin
        x = pyjl([1, 2, 3, 4, 5])
        @test pyeq(Bool, x.pop(), 5)
        @test pyjlvalue(x) == [1, 2, 3, 4]
        @test pyeq(Bool, x.pop(0), 1)
        @test pyjlvalue(x) == [2, 3, 4]
        @test pyeq(Bool, x.pop(1), 3)
        @test pyjlvalue(x) == [2, 4]
        @test pyeq(Bool, x.pop(-2), 2)
        @test pyjlvalue(x) == [4]
        @test_throws PyException x.pop(10)
    end
    @testset "remove" begin
        x = pyjl([1, 3, 2, 4, 5, 3, 1])
        @test pyjlvalue(x) == [1, 3, 2, 4, 5, 3, 1]
        x.remove(3)
        @test pyjlvalue(x) == [1, 2, 4, 5, 3, 1]
        @test_throws PyException x.remove(0)
        @test_throws PyException x.remove(nothing)
        @test_throws PyException x.remove("2")
        @test pyjlvalue(x) == [1, 2, 4, 5, 3, 1]
    end
    @testset "index" begin
        x = pyjl([1, 3, 2, 4, 5, 2, 1])
        @test pyeq(Bool, x.index(1), 0)
        @test pyeq(Bool, x.index(2), 2)
        @test pyeq(Bool, x.index(3), 1)
        @test pyeq(Bool, x.index(4), 3)
        @test pyeq(Bool, x.index(5), 4)
        @test pyeq(Bool, x.index(2.0), 2)
        @test_throws PyException x.index(0)
        @test_throws PyException x.index(6)
        @test_throws PyException x.index(nothing)
        @test_throws PyException x.index("2")
    end
    @testset "count" begin
        x = pyjl([1, 2, 3, 4, 5, 1, 2, 3, 1])
        @test pyeq(Bool, x.count(0), 0)
        @test pyeq(Bool, x.count(1), 3)
        @test pyeq(Bool, x.count(2), 2)
        @test pyeq(Bool, x.count(3), 2)
        @test pyeq(Bool, x.count(4), 1)
        @test pyeq(Bool, x.count(5), 1)
        @test pyeq(Bool, x.count(2.0), 2)
        @test pyeq(Bool, x.count(nothing), 0)
        @test pyeq(Bool, x.count("2"), 0)
    end

    @testset "PyObjectArray" begin
        # https://github.com/JuliaPy/PythonCall.jl/issues/543
        # Here we check the finalizer does not error
        # We must not reuse `arr` in this code once we finalize it!
        let arr = PyObjectArray([1, 2, 3])
            finalize(arr)
        end
    end
end
