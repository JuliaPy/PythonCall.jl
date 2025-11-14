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
    Base.iterate(x::Foo, st::Int = 1) = st <= x.value ? (st, st + 1) : nothing
    @testset "type" begin
        @test pyis(pytype(pyjl(Foo(1))), PythonCall.pyjlanytype)
        @test pyis(pytype(pyjl(nothing)), PythonCall.pyjlanytype)
        @test pyis(pytype(pyjl(missing)), PythonCall.pyjlanytype)
    end
    @testset "bool" begin
        @test pytruth(pyjl(true))
        @test !pytruth(pyjl(false))
        @test_throws Exception pytruth(pyjl(Foo(0)))
        @test_throws Exception pytruth(pyjl(Foo(1)))
        @test_throws Exception pytruth(pyjl(nothing))
        @test_throws Exception pytruth(pyjl(missing))
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
        @test pycontains(pydir(pyjl(Base)), "+")
        @test pycontains(pydir(pyjl(Base)), "Type")
        @test pycontains(pydir(pyjl(Base)), "nfields")
    end
    @testset "call" begin
        z = pyjl(Foo(1))(4, 5)
        @test pyconvert(String, z) == "1((4, 5))0"
        z = pyjl(Foo(1))(4, 5; foo = true, bar = true)
        @test pyconvert(String, z) == "1((4, 5))2"
    end
    @testset "callback" begin
        z = pyjl(Foo(1)).jl_callback(4, 5)
        @test pyconvert(String, z) == "1((<py 4>, <py 5>))0"
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
        z = pyjl(Foo(2)).__radd__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 + 2"
    end
    @testset "sub" begin
        z = pyjl(Foo(1)) - pyjl(Foo(2))
        @test pyconvert(String, z) == "1 - 2"
    end
    @testset "rsub" begin
        z = pyjl(Foo(2)).__rsub__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 - 2"
    end
    @testset "mul" begin
        z = pyjl(Foo(1)) * pyjl(Foo(2))
        @test pyconvert(String, z) == "1 * 2"
    end
    @testset "rmul" begin
        z = pyjl(Foo(2)).__rmul__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 * 2"
    end
    @testset "truediv" begin
        z = pyjl(Foo(1)) / pyjl(Foo(2))
        @test pyconvert(String, z) == "1 / 2"
    end
    @testset "rtruediv" begin
        z = pyjl(Foo(2)).__rtruediv__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 / 2"
    end
    @testset "floordiv" begin
        z = pyjl(Foo(1)) ÷ pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ÷ 2"
    end
    @testset "rfloordiv" begin
        z = pyjl(Foo(2)).__rfloordiv__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 ÷ 2"
    end
    @testset "mod" begin
        z = pyjl(Foo(1)) % pyjl(Foo(2))
        @test pyconvert(String, z) == "1 % 2"
    end
    @testset "rmod" begin
        z = pyjl(Foo(2)).__rmod__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 % 2"
    end
    @testset "pow" begin
        z = pyjl(Foo(1))^pyjl(Foo(2))
        @test pyconvert(String, z) == "1 ^ 2"
    end
    @testset "rpow" begin
        z = pyjl(Foo(2)).__rpow__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 ^ 2"
    end
    @testset "lshift" begin
        z = pyjl(Foo(1)) << pyjl(Foo(2))
        @test pyconvert(String, z) == "1 << 2"
    end
    @testset "rlshift" begin
        z = pyjl(Foo(2)).__rlshift__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 << 2"
    end
    @testset "rshift" begin
        z = pyjl(Foo(1)) >> pyjl(Foo(2))
        @test pyconvert(String, z) == "1 >> 2"
    end
    @testset "rrshift" begin
        z = pyjl(Foo(2)).__rrshift__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 >> 2"
    end
    @testset "and" begin
        z = pyjl(Foo(1)) & pyjl(Foo(2))
        @test pyconvert(String, z) == "1 & 2"
    end
    @testset "rand" begin
        z = pyjl(Foo(2)).__rand__(pyjl(Foo(1)))
        @test pyconvert(String, z) == "1 & 2"
    end
    @testset "or" begin
        z = pyjl(Foo(1)) | pyjl(Foo(2))
        @test pyconvert(String, z) == "1 | 2"
    end
    @testset "ror" begin
        z = pyjl(Foo(2)).__ror__(pyjl(Foo(1)))
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
        pyjl(Foo(1)).jl_display()
        pyjl(Foo(1)).jl_display(mime = "text/plain")
    end
    @testset "help" begin
        using REPL
        pyjl(Foo(1)).jl_help()
        pyjl(Foo(1)).jl_help(mime = "text/plain")
    end
    @testset "eval" begin
        m = pyjl(Main)
        # Basic behavior
        z = m.jl_eval("1 + 1")
        @test pyisjl(z)
        @test pyconvert(Any, z) === 2
        # Trailing whitespace
        z = m.jl_eval("1 + 2\n ")
        @test pyisjl(z)
        @test pyconvert(Any, z) === 3
    end
    @testset "to_py $x" for x in [1, 2.3, nothing, "foo"]
        y = Py(x)
        z = pyjl(x).jl_to_py()
        @test pyeq(Bool, pytype(z), pytype(y))
        @test pyeq(Bool, z, y)
    end
    @testset "iter" begin
        z = pylist(pyjl(Foo(3)))
        @test all(pyisjl, z)
        @test pyeq(Bool, z, pylist([pyjl(1), pyjl(2), pyjl(3)]))
    end
    @testset "next" begin
        z = pynext(pyjl(Foo(3)))
        @test pyisjl(z)
        @test pyeq(Bool, z, pyjl(1))
    end
    @testset "reversed" begin
        x = pyjl([1, 2, 3])
        y = x.__reversed__()
        @test pyisjl(y)
        @test pyjlvalue(y) == [3, 2, 1]
        @test pyjlvalue(x) == [1, 2, 3]
    end
    @testset "int" begin
        x = pyjl(34.0)
        y = x.__int__()
        @test pyisinstance(y, pybuiltins.int)
        @test pyeq(Bool, y, 34)
    end
    @testset "float" begin
        x = pyjl(12)
        y = x.__float__()
        @test pyisinstance(y, pybuiltins.float)
        @test pyeq(Bool, y, 12.0)
    end
    @testset "complex" begin
        x = pyjl(Complex(1, 2))
        y = x.__complex__()
        @test pyisinstance(y, pybuiltins.complex)
        @test pyeq(Bool, y, pycomplex(1, 2))
    end
    @testset "index" begin
        y = pyjl(12).__index__()
        @test pyisinstance(y, pybuiltins.int)
        @test pyeq(Bool, y, 12)
        @test_throws PyException pyjl(12.0).__index__()
    end
    @testset "trunc $x" for (x, y) in [(1.1, 1), (8.9, 8), (-1.3, -1), (-7.8, -7)]
        z = pyjl(x).__trunc__()
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, y)
    end
    @testset "floor $x" for (x, y) in [(1.1, 1), (8.9, 8), (-1.3, -2), (-7.8, -8)]
        z = pyjl(x).__floor__()
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, y)
    end
    @testset "ceil $x" for (x, y) in [(1.1, 2), (8.9, 9), (-1.3, -1), (-7.8, -7)]
        z = pyjl(x).__ceil__()
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, y)
    end
    @testset "round $x" for (x, y) in [(1.1, 1), (8.9, 9), (-1.3, -1), (-7.8, -8)]
        z = pyjl(x).__round__()
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, y)
    end
end

@testitem "collection" begin
    cases = [
        (x = fill(nothing), type = :array, list = pylist([nothing])),
        (x = [1, 2, 3], type = :vector, list = pylist([1, 2, 3])),
        (x = [1 2; 3 4], type = :array, list = pylist([1, 3, 2, 4])),
        (x = Set([1, 2, 3]), type = :set, list = pylist(Set([1, 2, 3]))),
        (x = Dict(1 => 2, 3 => 4), type = :dict, list = pylist(keys(Dict(1 => 2, 3 => 4)))),
        (x = keys(Dict()), type = :set, list = pylist()),
        (x = values(Dict()), type = :collection, list = pylist()),
        (x = (1, 2, 3), type = :collection, list = pylist([1, 2, 3])),
        (x = (x = 1, y = 2), type = :collection, list = pylist([1, 2])),
        (x = Ref(nothing), type = :collection, list = pylist([nothing])),
    ]
    @testset "type $(c.x)" for c in cases
        y = pyjlcollection(c.x)
        @test pyisinstance(y, PythonCall.JlWrap.pyjlcollectiontype)
        @test pyis(pytype(y), getproperty(PythonCall.JlWrap, Symbol(:pyjl, c.type, :type)))
    end
    @testset "len $(c.x)" for c in cases
        @test pylen(pyjlcollection(c.x)) == length(c.x)
    end
    @testset "bool $(c.x)" for c in cases
        @test pytruth(pyjlcollection(c.x)) == !isempty(c.x)
    end
    @testset "iter $(c.x)" for c in cases
        @test pyeq(Bool, pylist(pyjlcollection(c.x)), c.list)
    end
    @testset "hash $(c.x)" for c in cases
        # not sure why but the bottom bits don't always match
        @test mod(pyhash(pyjlcollection(c.x)), UInt32) >> 16 == mod(hash(c.x), UInt32) >> 16
    end
    @testset "eq $(c1.x) $(c2.x)" for (c1, c2) in Iterators.product(cases, cases)
        @test pyeq(Bool, pyjlcollection(c1.x), pyjlcollection(c2.x)) == (c1.x == c2.x)
    end
    @testset "contains $(c.x) $(v)" for (c, v) in Iterators.product(
        cases,
        [nothing, 0, 1, 2, 3, 4, 5, 0.0, 0.5, 1.0],
    )
        if !isa(c.x, Dict)
            @test pycontains(pyjlcollection(c.x), v) == (v in c.x)
        end
    end
    @testset "copy $(c.x)" for c in cases
        copyable = try
            copy(c.x)
            true
        catch
            false
        end
        y = pyjlcollection(c.x)
        if copyable
            z = y.copy()
            @test pyis(pytype(y), pytype(z))
            yv = pyjlvalue(y)
            zv = pyjlvalue(z)
            @test yv === c.x
            @test yv == zv
            @test yv !== zv
        else
            @test_throws PyException y.copy()
        end
    end
    @testset "clear $(c.x)" for c in cases
        # make a copy or skip the test
        x2 = try
            copy(c.x)
        catch
            continue
        end
        len = length(x2)
        # see if the collection can be emptied
        clearable = try
            empty!(copy(c.x))
            true
        catch
            false
        end
        # try clearing the collection
        y = pyjlcollection(x2)
        @test pylen(y) == len
        if clearable
            y.clear()
            @test pylen(y) == 0
            @test length(x2) == 0
        else
            @test_throws PyException y.clear()
            @test pylen(y) == len
            @test length(x2) == len
        end
    end
end

@testitem "array" setup=[Setup] begin
    @testset "type" begin
        @test pyis(pytype(pyjlarray(fill(nothing))), PythonCall.pyjlarraytype)
        @test pyis(pytype(pyjlarray([1, 2, 3])), PythonCall.pyjlvectortype)
        @test pyis(pytype(pyjlarray([1 2; 3 4])), PythonCall.pyjlarraytype)
    end
    @testset "bool" begin
        @test !pytruth(pyjlarray(fill(nothing, 0, 1)))
        @test !pytruth(pyjlarray(fill(nothing, 1, 0)))
        @test pytruth(pyjlarray(fill(nothing)))
        @test pytruth(pyjlarray(fill(nothing, 1, 2)))
        @test pytruth(pyjlarray(fill(nothing, 1, 2, 3)))
    end
    @testset "ndim" begin
        @test pyeq(Bool, pyjlarray(fill(nothing)).ndim, 0)
        @test pyeq(Bool, pyjlarray(fill(nothing, 1)).ndim, 1)
        @test pyeq(Bool, pyjlarray(fill(nothing, 1, 1)).ndim, 2)
        @test pyeq(Bool, pyjlarray(fill(nothing, 1, 1, 1)).ndim, 3)
    end
    @testset "shape" begin
        @test pyeq(Bool, pyjlarray(fill(nothing)).shape, ())
        @test pyeq(Bool, pyjlarray(fill(nothing, 3)).shape, (3,))
        @test pyeq(Bool, pyjlarray(fill(nothing, 3, 5)).shape, (3, 5))
        @test pyeq(Bool, pyjlarray(fill(nothing, 3, 5, 2)).shape, (3, 5, 2))
    end
    @testset "getitem" begin
        x = pyjlarray([1, 2, 3, 4, 5])
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
        x = pyjlarray([1 2; 3 4])
        @test pyeq(Bool, x[0, 0], 1)
        @test pyeq(Bool, x[0, 1], 2)
        @test pyeq(Bool, x[1, 0], 3)
        @test pyeq(Bool, x[1, 1], 4)
        @test pyjlvalue(x[1, pyslice(nothing)]) == [3, 4]
        @test pyjlvalue(x[pyslice(nothing), 1]) == [2, 4]
    end
    @testset "setitem" begin
        x = [0 0; 0 0]
        y = pyjlarray(x)
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
        y = pyjlarray(x)
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
        x = pyjlarray([1, 2, 3, 4, 5, 6, 7, 8])
        @test pyeq(Bool, x.shape, (8,))
        y = x.reshape((2, 4))
        @test pyeq(Bool, y.shape, (2, 4))
        @test pyjlvalue(y) == [1 3 5 7; 2 4 6 8]
    end
    @testset "copy" begin
        x = pyjlarray([1 2; 3 4])
        y = x.copy()
        @test pyis(pytype(y), PythonCall.pyjlarraytype)
        @test pyjlvalue(x) == pyjlvalue(y)
        @test typeof(pyjlvalue(x)) == typeof(pyjlvalue(y))
        @test pyjlvalue(x) !== pyjlvalue(y)
        x[0, 0] = 0
        @test pyjlvalue(x) == [0 2; 3 4]
        @test pyjlvalue(y) == [1 2; 3 4]
    end
    @testset "__array__" begin
        if Setup.devdeps
            np = pyimport("numpy")

            numeric = pyjlarray(Float64[1, 2, 3])
            numeric_array = numeric.__array__()
            @test pyisinstance(numeric_array, np.ndarray)
            @test pyconvert(Vector{Float64}, numeric_array) == [1.0, 2.0, 3.0]

            numeric_no_copy = numeric.__array__(copy=false)
            numeric_data = pyjlvalue(numeric)
            numeric_data[1] = 42.0
            @test pyconvert(Vector{Float64}, numeric_no_copy) == [42.0, 2.0, 3.0]

            string_array = pyjlarray(["a", "b"])
            string_result = string_array.__array__()
            @test pyisinstance(string_result, np.ndarray)
            @test pyconvert(Vector{String}, pybuiltins.list(string_result)) == ["a", "b"]

            err = try
                string_array.__array__(copy=false)
                nothing
            catch err
                err
            end
            @test err !== nothing
            @test err isa PythonCall.PyException
            @test pyis(err._t, pybuiltins.ValueError)
            @test occursin(
                "copy=False is not supported when collecting ArrayValue data",
                sprint(showerror, err),
            )
        else
            @test_skip Setup.devdeps
        end
    end
    @testset "array_interface" begin
        x = pyjlarray(Float32[1 2 3; 4 5 6]).__array_interface__
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
        m = pybuiltins.memoryview(pyjlarray(Float32[1 2 3; 4 5 6]))
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
    @testset "iter" begin
        @test pyeq(Bool, pybuiltins.list(pyjlarray(fill(0))), pylist([0]))
        @test pyeq(Bool, pybuiltins.list(pyjlarray([1, 2, 3])), pylist([1, 2, 3]))
        @test pyeq(Bool, pybuiltins.list(pyjlarray([1 2; 3 4])), pylist([1, 3, 2, 4]))
    end
    @testset "eq" begin
        @test pyeq(Bool, pyjlarray([1, 2, 3]), pyjlarray([1, 2, 3]))
        @test pyeq(Bool, pyjlarray([1, 2, 3]), pyjlarray(1:3))
        @test !pyeq(Bool, pyjlarray([1, 2, 3]), pyjlarray([2, 3, 4]))
        @test !pyeq(Bool, pyjlarray([1, 2, 3]), pylist([1, 2, 3]))
    end
    @testset "hash" begin
        @test pyhash(pyjlarray([1, 2, 3])) == pyhash(pyjlarray([1, 2, 3]))
        @test pyhash(pyjlarray([1, 2, 3])) == pyhash(pyjlarray(1:3))
        @test pyhash(pyjlarray([1, 2, 3])) != pyhash(pyjlarray([2, 3, 4]))
        @test pyhash(pyjlarray([1, 2, 3])) != pyhash(pytuple([1, 2, 3]))
    end
    @testset "len" begin
        @test pylen(pyjlarray([1, 2, 3])) == 3
        @test pylen(pyjlarray([])) == 0
        @test pylen(pyjlarray(fill(0))) == 1
        @test pylen(pyjlarray([1 2; 3 4])) == 4
    end
end

@testitem "base" begin

end

@testitem "callback" begin

end

@testitem "dict" begin
    @testset "type" begin
        @test pyis(pytype(pyjldict(Dict())), PythonCall.pyjldicttype)
    end
    @testset "bool" begin
        @test !pytruth(pyjldict(Dict()))
        @test pytruth(pyjldict(Dict("one" => 1, "two" => 2)))
    end
    @testset "iter" begin
        @test pyeq(Bool, pyset(pyjldict(Dict())), pyset())
        @test pyeq(Bool, pyset(pyjldict(Dict(1 => 2, 3 => 4))), pyset([1, 3]))
    end
    @testset "contains" begin
        x = pyjldict(Dict(1 => 2, 3 => 4))
        @test pycontains(x, 1)
        @test !pycontains(x, 2)
        @test pycontains(x, 3)
        @test !pycontains(x, 4)
        @test !pycontains(x, 1.2)
        @test pycontains(x, 1.0)
        @test !pycontains(x, nothing)
    end
    @testset "getitem" begin
        x = pyjldict(Dict(1 => 2, 3 => 4))
        @test pyisinstance(x[1], pybuiltins.int)
        @test pyeq(Bool, x[1], 2)
        @test pyisinstance(x[3], pybuiltins.int)
        @test pyeq(Bool, x[3], 4)
        @test_throws PyException x[2]
    end
    @testset "setitem" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        y[1] = pyint(11)
        @test x[1] === 11
        y[2] = pyfloat(22.0)
        @test x[2] === 22
    end
    @testset "delitem" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        pydelitem(y, 3)
        @test x == Dict(1 => 2)
        pydelitem(y, 1)
        @test x == Dict()
    end
    @testset "keys" begin
        x = pyjldict(Dict(1 => 2, 3 => 4)).keys()
        @test all(pyisinstance(k, pybuiltins.int) for k in x)
        @test pyeq(Bool, pyset(x), pyset([1, 3]))
    end
    @testset "values" begin
        x = pyjldict(Dict(1 => 2, 3 => 4)).values()
        @test all(pyisinstance(k, pybuiltins.int) for k in x)
        @test pyeq(Bool, pyset(x), pyset([2, 4]))
    end
    @testset "items" begin
        x = pyjldict(Dict(1 => 2, 3 => 4)).items()
        @test all(pyisinstance(i, pybuiltins.tuple) for i in x)
        @test all(pylen(i) == 2 for i in x)
        @test all(pyisinstance(i[0], pybuiltins.int) for i in x)
        @test all(pyisinstance(i[1], pybuiltins.int) for i in x)
        @test pyeq(Bool, pyset(x), pyset([(1, 2), (3, 4)]))
    end
    @testset "get" begin
        x = pyjldict(Dict(1 => 2, 3 => 4))
        y = x.get(1)
        @test pyisinstance(y, pybuiltins.int)
        @test pyeq(Bool, y, 2)
        y = x.get(3)
        @test pyisinstance(y, pybuiltins.int)
        @test pyeq(Bool, y, 4)
        y = x.get(5)
        @test pyis(y, pybuiltins.None)
        y = x.get(5, 0)
        @test pyisinstance(y, pybuiltins.int)
        @test pyeq(Bool, y, 0)
    end
    @testset "setdefault" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        z = y.setdefault(1, 0)
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, 2)
        @test x == Dict(1 => 2, 3 => 4)
        z = y.setdefault(2, 0)
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, 0)
        @test x == Dict(1 => 2, 3 => 4, 2 => 0)
        z = y.setdefault(2, 99)
        @test pyisinstance(z, pybuiltins.int)
        @test pyeq(Bool, z, 0)
        @test x == Dict(1 => 2, 3 => 4, 2 => 0)
    end
    @testset "pop" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        z1 = y.pop(1)
        @test pyisinstance(z1, pybuiltins.int)
        @test pyeq(Bool, z1, 2)
        @test x == Dict(3 => 4)
        @test_throws PyException y.pop(2)
        z2 = y.pop(3)
        @test pyisinstance(z2, pybuiltins.int)
        @test pyeq(Bool, z2, 4)
        @test x == Dict()
    end
    @testset "popitem" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        z1 = y.popitem()
        z2 = y.popitem()
        @test all(pyisinstance(z, pybuiltins.tuple) for z in [z1, z2])
        @test all(pylen(z) == 2 for z in [z1, z2])
        @test all(pyisinstance(z[0], pybuiltins.int) for z in [z1, z2])
        @test all(pyisinstance(z[1], pybuiltins.int) for z in [z1, z2])
        @test pyeq(Bool, pyset([z1, z2]), pyset([(1, 2), (3, 4)]))
    end
    @testset "update" begin
        x = Dict(1 => 2, 3 => 4)
        y = pyjldict(x)
        y.update(pydict([(3, 3.0), (2, 2.0)]))
        @test x == Dict(1 => 2, 2 => 2, 3 => 3)
    end
end

@testitem "io/base" begin
    @testset "close" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        @test isopen(x)
        y.close()
        @test !isopen(x)
    end
    @testset "closed" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        z = y.closed
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, false)
        close(x)
        z = y.closed
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, true)
    end
    @testset "fileno" begin
        # IOBuffer has no fileno
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        @test_throws PyException y.fileno()
        # check some file that has a fileno
        mktemp() do name, x
            y = pytextio(x)
            z = y.fileno()
            @test pyisinstance(z, pybuiltins.int)
            @test pyconvert(Int, z) == Base.cconvert(Cint, fd(x))
        end
    end
    @testset "flush" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        y.flush()
        # TODO: check it actually flushed something
    end
    @testset "isatty $(typeof(x))" for (x, y) in [
        (IOBuffer(), false),
        (devnull, false),
        (stdout, stdout isa Base.TTY),
    ]
        # TODO: how to get a TTY in a test environment??
        z = pytextio(x).isatty()
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, y)
    end
    @testset "readable $(typeof(x))" for (x, y) in [
        (IOBuffer(), true),
        (devnull, false),
        (stdin, true),
    ]
        z = pytextio(x).readable()
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, y)
    end
    @testset "readlines" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pytextio(x)
        z = y.readlines()
        @test pyisinstance(z, pybuiltins.list)
        @test pylen(z) == 2
        @test all(pyisinstance(line, pybuiltins.str) for line in z)
        @test pyeq(Bool, z, pylist(["hello\n", "world\n"]))
    end
    @testset "seek" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        @test position(x) == 0
        zs = Py[]
        push!(zs, y.seek(1))
        @test position(x) == 1
        push!(zs, y.seek(2, 0))
        @test position(x) == 2
        push!(zs, y.seek(1, 1))
        @test position(x) == 3
        push!(zs, y.seek(-2, 2))
        @test position(x) == 4
        @test all(pyisinstance(z, pybuiltins.int) for z in zs)
        @test pyeq(Bool, pylist(zs), pylist([1, 2, 3, 4]))
    end
    @testset "seekable $(typeof(x))" for (x, y) in [
        (IOBuffer(), true),
        (devnull, true),
        (stdin, true),
        (stdout, true),
    ]
        z = pytextio(x).seekable()
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, y)
    end
    @testset "tell" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        zs = Py[]
        @test position(x) == 0
        push!(zs, y.tell())
        seek(x, 5)
        push!(zs, y.tell())
        @test all(pyisinstance(z, pybuiltins.int) for z in zs)
        @test pyeq(Bool, pylist(zs), pylist([0, 5]))
    end
    @testset "truncate" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        y.truncate(5)
        seekend(x)
        @test position(x) == 5
        seek(x, 3)
        y.truncate()
        seekstart(x)
        seekend(x)
        @test position(x) == 3
    end
    @testset "writable $(typeof(x))" for (x, y) in [
        (IOBuffer(), true),
        (IOBuffer(""), false),
        (devnull, true),
        (stdout, true),
    ]
        z = pytextio(x).writable()
        @test pyisinstance(z, pybuiltins.bool)
        @test pyeq(Bool, z, y)
    end
    @testset "writelines" begin
        x = IOBuffer()
        y = pytextio(x)
        y.writelines(pylist(["test\n", "message\n"]))
        seekstart(x)
        @test readline(x) == "test"
        @test readline(x) == "message"
        @test readline(x) == ""
    end
    @testset "enter/exit" begin
        x = IOBuffer()
        seekstart(x)
        y = pytextio(x)
        @test isopen(x)
        r = pywith(y) do z
            @test pyis(z, y)
            12
        end
        @test r === 12
        @test !isopen(x)
        # same again by cause an error
        x = IOBuffer()
        seekstart(x)
        y = pytextio(x)
        @test isopen(x)
        @test_throws PyException pywith(y) do z
            z.invalid_attr
        end
        @test !isopen(x)  # should still get closed
    end
    @testset "iter" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pytextio(x)
        zs = pylist(y)
        @test all(pyisinstance(z, pybuiltins.str) for z in zs)
        @test pyeq(Bool, zs, pylist(["hello\n", "world\n"]))
    end
    @testset "next" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pytextio(x)
        zs = Py[]
        push!(zs, y.__next__())
        push!(zs, y.__next__())
        @test_throws PyException y.__next__()
        @test all(pyisinstance(z, pybuiltins.str) for z in zs)
        @test pyeq(Bool, pylist(zs), pylist(["hello\n", "world\n"]))
    end
end

@testitem "io/binary" begin
    @testset "type" begin
        @test pyis(pytype(pybinaryio(devnull)), PythonCall.pyjlbinaryiotype)
    end
    @testset "bool" begin
        @test pytruth(pybinaryio(devnull))
    end
    @testset "detach" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pybinaryio(x)
        @test_throws PyException y.detach()
    end
    @testset "read" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pybinaryio(x)
        z = y.read()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b"hello\nworld\n"))
        z = y.read()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b""))
    end
    @testset "read1" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pybinaryio(x)
        z = y.read1()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b"hello\nworld\n"))
        z = y.read1()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b""))
    end
    @testset "readline" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pybinaryio(x)
        z = y.readline()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b"hello\n"))
        z = y.readline()
        @test pyisinstance(z, pybuiltins.bytes)
        @test pyeq(Bool, z, pybytes(b""))
    end
    @testset "readinto" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pybinaryio(x)
        z = pybuiltins.bytearray(pybytes(b"xxxxxxxxxx"))
        n = y.readinto(z)
        @test pyisinstance(n, pybuiltins.int)
        @test pyeq(Bool, n, 6)
        @test pyeq(Bool, pybytes(z), pybytes(b"hello\nxxxx"))
    end
    @testset "readinto1" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pybinaryio(x)
        z = pybuiltins.bytearray(pybytes(b"xxxxxxxxxx"))
        n = y.readinto(z)
        @test pyisinstance(n, pybuiltins.int)
        @test pyeq(Bool, n, 6)
        @test pyeq(Bool, pybytes(z), pybytes(b"hello\nxxxx"))
    end
    @testset "write" begin
        x = IOBuffer()
        print(x, "hello\n")
        y = pybinaryio(x)
        y.write(pybytes(b"world\n"))
        @test String(take!(x)) == "hello\nworld\n"
    end
end

@testitem "io/text" begin
    @testset "type" begin
        @test pyis(pytype(pytextio(devnull)), PythonCall.pyjltextiotype)
    end
    @testset "bool" begin
        @test pytruth(pytextio(devnull))
    end
    @testset "encoding" begin
        x = IOBuffer()
        print(x, "hello\n")
        y = pytextio(x)
        z = y.encoding
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "UTF-8")
    end
    @testset "errors" begin
        x = IOBuffer()
        print(x, "hello\n")
        y = pytextio(x)
        z = y.errors
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "strict")
    end
    @testset "detach" begin
        x = IOBuffer()
        print(x, "hello\n")
        y = pytextio(x)
        @test_throws PyException y.detach()
    end
    @testset "read" begin
        x = IOBuffer()
        print(x, "hello\n")
        print(x, "world\n")
        seekstart(x)
        y = pytextio(x)
        z = y.read()
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "hello\nworld\n")
        z = y.read()
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "")
    end
    @testset "readline" begin
        x = IOBuffer()
        print(x, "hello\n")
        seekstart(x)
        y = pytextio(x)
        z = y.readline()
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "hello\n")
        z = y.readline()
        @test pyisinstance(z, pybuiltins.str)
        @test pyeq(Bool, z, "")
    end
    @testset "write" begin
        x = IOBuffer()
        print(x, "hello\n")
        y = pytextio(x)
        y.write("world!")
        @test String(take!(x)) == "hello\nworld!"
    end
end

@testitem "iter" begin
    x1 = [1, 2, 3, 4, 5]
    x2 = pyjl(x1)
    x3 = pylist(x2)
    x4 = pyconvert(Vector{Int}, x3)
    @test x1 == x4
end

@testitem "objectarray" begin

end

@testitem "set" begin
    @testset "type" begin
        @test pyis(pytype(pyjlset(Set())), PythonCall.pyjlsettype)
    end
    @testset "bool" begin
        @test !pytruth(pyjlset(Set()))
        @test pytruth(pyjlset(Set([1, 2, 3])))
    end
    @testset "add" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.add(1)
        @test x == Set([1, 2, 3])
        y.add(0)
        @test x == Set([0, 1, 2, 3])
        @test_throws PyException y.add(nothing)
    end
    @testset "discard" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.discard(1)
        @test x == Set([2, 3])
        y.discard(1)
        @test x == Set([2, 3])
    end
    @testset "pop" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        zs = Py[]
        for i = 1:3
            push!(zs, y.pop())
            @test length(x) == 3 - i
        end
        @test_throws PyException y.pop()
        @test all(pyisinstance(z, pybuiltins.int) for z in zs)
        @test pyeq(Bool, pyset(zs), pyset([1, 2, 3]))
    end
    @testset "remove" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.remove(1)
        @test x == Set([2, 3])
        @test_throws PyException y.remove(1)
    end
    @testset "difference" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.difference(pyset([1, 3, 5]))
        @test pyeq(Bool, z, pyset([2]))
    end
    @testset "intersection" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.intersection(pyset([1, 3, 5]))
        @test pyeq(Bool, z, pyset([1, 3]))
    end
    @testset "symmetric_difference" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.symmetric_difference(pyset([1, 3, 5]))
        @test pyeq(Bool, z, pyset([2, 5]))
    end
    @testset "union" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.union(pyset([1, 3, 5]))
        @test pyeq(Bool, z, pyset([1, 2, 3, 5]))
    end
    @testset "isdisjoint" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.isdisjoint(pyset([1, 3, 5]))
        @test pyeq(Bool, z, false)
        z = y.isdisjoint(pyset([0, 5]))
        @test pyeq(Bool, z, true)
    end
    @testset "issubset" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.issubset(pyset([1, 3, 5]))
        @test pyeq(Bool, z, false)
        z = y.issubset(pyset([1, 2, 3, 4, 5]))
        @test pyeq(Bool, z, true)
    end
    @testset "issuperset" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        z = y.issuperset(pyset([1, 3, 5]))
        @test pyeq(Bool, z, false)
        z = y.issuperset(pyset([1, 3]))
        @test pyeq(Bool, z, true)
    end
    @testset "difference_update" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.difference_update(pyset([1, 3, 5]))
        @test x == Set([2])
    end
    @testset "intersection_update" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.intersection_update(pyset([1, 3, 5]))
        @test x == Set([1, 3])
    end
    @testset "symmetric_difference_update" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.symmetric_difference_update(pyset([1, 3, 5]))
        @test x == Set([2, 5])
    end
    @testset "update" begin
        x = Set([1, 2, 3])
        y = pyjlset(x)
        y.update(pyset([1, 3, 5]))
        @test x == Set([1, 2, 3, 5])
    end
end

@testitem "vector" begin
    @testset "type" begin
        @test pyis(pytype(pyjlarray([1, 2, 3, 4])), PythonCall.pyjlvectortype)
    end
    @testset "bool" begin
        @test !pytruth(pyjlarray([]))
        @test pytruth(pyjlarray([1]))
        @test pytruth(pyjlarray([1, 2]))
    end
    @testset "resize" begin
        x = pyjlarray([1, 2, 3, 4, 5])
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
        x = pyjlarray([4, 6, 2, 3, 7, 6, 1])
        x.sort()
        @test pyjlvalue(x) == [1, 2, 3, 4, 6, 6, 7]
        x = pyjlarray([4, 6, 2, 3, 7, 6, 1])
        x.sort(reverse = true)
        @test pyjlvalue(x) == [7, 6, 6, 4, 3, 2, 1]
        x = pyjlarray([4, -6, 2, -3, 7, -6, 1])
        x.sort(key = abs)
        @test pyjlvalue(x) == [1, 2, -3, 4, -6, -6, 7]
        x = pyjlarray([4, -6, 2, -3, 7, -6, 1])
        x.sort(key = abs, reverse = true)
        @test pyjlvalue(x) == [7, -6, -6, 4, -3, 2, 1]
    end
    @testset "reverse" begin
        x = pyjlarray([1, 2, 3, 4, 5])
        x.reverse()
        @test pyjlvalue(x) == [5, 4, 3, 2, 1]
    end
    @testset "clear" begin
        x = pyjlarray([1, 2, 3, 4, 5])
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.clear()
        @test pyjlvalue(x) == []
    end
    @testset "reversed" begin
        x = pyjlarray([1, 2, 3, 4, 5])
        y = pybuiltins.reversed(x)
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        @test pyjlvalue(y) == [5, 4, 3, 2, 1]
    end
    @testset "insert" begin
        x = pyjlarray([1, 2, 3])
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
        x = pyjlarray([1, 2, 3])
        x.append(4)
        @test pyjlvalue(x) == [1, 2, 3, 4]
        x.append(5.0)
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        @test_throws PyException x.append(nothing)
        @test_throws PyException x.append(1.2)
        @test_throws PyException x.append("2")
    end
    @testset "extend" begin
        x = pyjlarray([1, 2, 3])
        x.extend(pylist())
        @test pyjlvalue(x) == [1, 2, 3]
        x.extend(pylist([4, 5]))
        @test pyjlvalue(x) == [1, 2, 3, 4, 5]
        x.extend(pylist([6.0]))
        @test pyjlvalue(x) == [1, 2, 3, 4, 5, 6]
    end
    @testset "pop" begin
        x = pyjlarray([1, 2, 3, 4, 5])
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
        x = pyjlarray([1, 3, 2, 4, 5, 3, 1])
        @test pyjlvalue(x) == [1, 3, 2, 4, 5, 3, 1]
        x.remove(3)
        @test pyjlvalue(x) == [1, 2, 4, 5, 3, 1]
        @test_throws PyException x.remove(0)
        @test_throws PyException x.remove(nothing)
        @test_throws PyException x.remove("2")
        @test pyjlvalue(x) == [1, 2, 4, 5, 3, 1]
    end
    @testset "index" begin
        x = pyjlarray([1, 3, 2, 4, 5, 2, 1])
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
        x = pyjlarray([1, 2, 3, 4, 5, 1, 2, 3, 1])
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
