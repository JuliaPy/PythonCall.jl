@testitem "PyArray" begin
    x = pyimport("array").array("i", pylist([1, 2, 3]))
    y = PyArray(x)
    z = PyArray{Cint,1,false,false,Cint}(x)
    @testset "construct" begin
        @test y isa PyArray{Cint,1,true,true,Cint}
        @test z isa PyArray{Cint,1,false,false,Cint}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "length" begin
        @test length(y) === 3
        @test length(z) === 3
    end
    @testset "size" begin
        @test size(y) === (3,)
        @test size(z) === (3,)
    end
    @testset "IndexStyle" begin
        @test Base.IndexStyle(y) === Base.IndexLinear()
        @test Base.IndexStyle(z) === Base.IndexCartesian()
    end
    @testset "strides" begin
        @test strides(y) === (1,)
        @test strides(z) === (1,)
    end
    @testset "elsize" begin
        @test Base.elsize(y) === sizeof(Cint)
        @test Base.elsize(z) === sizeof(Cint)
        @test Base.elsize(PyArray{Cint,1,true,true,Cint}) === sizeof(Cint)
        @test Base.elsize(PyArray{Cint,1,false,false,Cint}) === sizeof(Cint)
        @test_throws Exception elsize(PyArray{Cint,1,true,false,Cchar})
    end
    @testset "getindex" begin
        @test_throws BoundsError y[0]
        @test y[1] == 1
        @test y[2] == 2
        @test y[3] == 3
        @test_throws BoundsError y[4]
        @test_throws BoundsError z[0]
        @test z[1] == 1
        @test z[2] == 2
        @test z[3] == 3
        @test_throws BoundsError z[4]
    end
    @testset "copy" begin
        @test copy(y) == [1, 2, 3]
        @test copy(z) == [1, 2, 3]
    end
    @testset "setindex!" begin
        @test_throws BoundsError y[0] = 0
        y[2] = 22
        @test y[2] == 22
        y[2] = 2
        @test y[2] == 2
        @test_throws BoundsError y[4] = 0
        @test_throws Exception z[0] = 0
        @test_throws Exception z[1] = 0
        @test_throws Exception z[2] = 0
        @test_throws Exception z[3] = 0
        @test_throws Exception z[4] = 0
    end
    @testset "mutate" begin
        y[2] = 22
        @test pyconvert(Int, x[1]) == 22
        @test y[2] == 22
        @test z[2] == 22
        y[2] = 2
        x[2] = 33
        @test pyconvert(Int, x[2]) == 33
        @test y[3] == 33
        @test z[3] == 33
        x[2] = 3
        @test y == [1, 2, 3]
        @test z == [1, 2, 3]
    end
    @testset "serialize" begin
        using Serialization: serialize, deserialize
        io = IOBuffer()
        serialize(io, y)
        seekstart(io)
        y2 = deserialize(io)
        @test typeof(y) == typeof(y2)
        @test eltype(y) == eltype(y2)
        @test ndims(y) == ndims(y2)
        @test size(y) == size(y2)
        @test strides(y) == strides(y2)
        @test y == y2
    end
end

@testitem "PyDict" begin
    x = pydict(["foo" => 12])
    y = PyDict(x)
    z = PyDict{String,Int}(x)
    @testset "construct" begin
        @test y isa PyDict{Py,Py}
        @test PyDict{String}(x) isa PyDict{String,Py}
        @test z isa PyDict{String,Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "length" begin
        @test length(y) == 1
        @test length(z) == 1
    end
    @testset "copy" begin
        t = copy(z)
        @test t isa PyDict{String,Int}
        @test !pyis(t, z)
        t["bar"] = 34
        @test z == Dict("foo" => 12)
        @test t == Dict("foo" => 12, "bar" => 34)
    end
    @testset "iterate" begin
        @test collect(z) == ["foo" => 12]
    end
    @testset "iterate keys" begin
        @test collect(keys(z)) == ["foo"]
    end
    @testset "getindex" begin
        @test z["foo"] === 12
        @test_throws KeyError z["bar"]
    end
    @testset "setindex!" begin
        t = copy(z)
        @test setindex!(t, 34, "bar") === t
        @test t == Dict("foo" => 12, "bar" => 34)
        @test setindex!(t, 56, "foo") === t
        @test t == Dict("foo" => 56, "bar" => 34)
        @test_throws Exception setindex!(t, 0, nothing)
        @test_throws Exception setindex!(t, nothing, "foo")
        @test t == Dict("foo" => 56, "bar" => 34)
    end
    @testset "delete!" begin
        t = copy(z)
        @test delete!(t, "bar") === t
        @test t == Dict("foo" => 12)
        @test delete!(t, 0) === t
        @test t == Dict("foo" => 12)
        @test delete!(t, "foo") === t
        @test isempty(t)
        @test delete!(t, "foo") === t
        @test isempty(t)
    end
    @testset "empty!" begin
        t = copy(z)
        @test !isempty(t)
        @test empty!(t) === t
        @test isempty(t)
    end
    @testset "haskey" begin
        @test haskey(z, "foo")
        @test !haskey(z, "bar")
        @test !haskey(z, nothing)
        @test !haskey(z, 99)
    end
    @testset "get" begin
        t = copy(z)
        @test get(t, "foo", nothing) === 12
        @test get(t, "bar", nothing) === nothing
        @test get(t, nothing, missing) === missing
        @test get(t, 0, 1) === 1
        @test get(Vector, t, "foo") === 12
        @test get(Vector, t, "bar") == []
        @test t == Dict("foo" => 12)
    end
    @testset "get!" begin
        t = copy(z)
        @test get!(t, "foo", 0) === 12
        @test t == Dict("foo" => 12)
        @test get!(t, "bar", 0) === 0
        @test t == Dict("foo" => 12, "bar" => 0)
        @test get!(() -> 99, t, "foo") === 12
        @test t == Dict("foo" => 12, "bar" => 0)
        @test get!(() -> 99, t, "baz") === 99
        @test t == Dict("foo" => 12, "bar" => 0, "baz" => 99)
        @test_throws Exception get!(t, 0, 0)
        @test_throws Exception get!(t, "", "")
        @test_throws Exception get!(() -> 99, t, 0)
        @test_throws Exception get!(Vector, t, "")
        @test t == Dict("foo" => 12, "bar" => 0, "baz" => 99)
    end
    @testset "construct empty" begin
        @test PyDict() isa PyDict{Py,Py}
        @test PyDict{String}() isa PyDict{String,Py}
        @test PyDict{String,Int}() isa PyDict{String,Int}
        @test isempty(PyDict{String,Int}())
    end
end

@testitem "PyIO" begin
    b0 = pyimport("io").BytesIO()
    s0 = pyimport("io").StringIO()
    b = PyIO(b0)
    s = PyIO(s0)
    @testset "ispy" begin
        @test ispy(b)
        @test ispy(s)
    end
    @testset "Py" begin
        @test Py(b) === b0
        @test Py(s) === s0
    end
    @testset "test" begin
        @test !b.text
        @test s.text
    end
    @testset "convert" begin
        ss = pyconvert(PyIO, s)
        bb = pyconvert(PyIO, b)
        @test ss isa PyIO
        @test bb isa PyIO
        @test Py(ss) === s0
        @test Py(bb) === b0
    end
    @testset "io" for io in [b, s]
        @test eof(io)
        @test position(io) == 0
        @test write(io, "hello") == 5
        @test position(io) == 5
        @test eof(io)
        seekstart(io)
        flush(io)
        @test position(io) == 0
        @test !eof(io)
        flush(io)
        @test read(io, String) == "hello"
        @test eof(io)
        seekstart(io)
        @test position(io) == 0
        seekend(io)
        @test position(io) == 5
        @test eof(io)
        seek(io, 0)
        @test position(io) == 0
        @test !eof(io)
        seek(io, 3)
        @test position(io) == 3
        @test !eof(io)
        seek(io, 5)
        @test position(io) == 5
        @test eof(io)
        seekstart(io)
        truncate(io, 3)
        @test position(io) == 0
        @test read(io, String) == "hel"
        @test position(io) == 3
        @test eof(io)
    end
    @testset "isreadable" begin
        @test isreadable(b)
        @test isreadable(s)
    end
    @testset "iswritable" begin
        @test iswritable(b)
        @test iswritable(s)
    end
    @testset "isopen" begin
        @test isopen(b)
        @test isopen(s)
    end
    @testset "close" begin
        close(b)
        close(s)
        @test !isopen(b)
        @test !isopen(s)
    end
end

@testitem "PyIterable" begin
    x = pylist([1, 2, 3])
    y = PyIterable(x)
    z = PyIterable{Int}(x)
    @testset "construct" begin
        @test y isa PyIterable{Py}
        @test z isa PyIterable{Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "iterate" begin
        @test Base.IteratorSize(typeof(y)) === Base.SizeUnknown()
        @test Base.IteratorSize(typeof(z)) === Base.SizeUnknown()
        @test eltype(y) == Py
        @test eltype(z) == Int
        @test collect(z) == [1, 2, 3]
    end
end

@testitem "PyList" begin
    x = pylist([1, 2, 3])
    y = PyList(x)
    z = PyList{Int}(x)
    @testset "construct" begin
        @test y isa PyList{Py}
        @test z isa PyList{Int}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test Py(y) === x
        @test Py(z) === x
    end
    @testset "length" begin
        @test length(y) == 3
        @test length(z) == 3
    end
    @testset "size" begin
        @test size(y) == (3,)
        @test size(z) == (3,)
    end
    @testset "getindex" begin
        @test y[1] isa Py
        @test pyeq(Bool, y[1], 1)
        @test pyeq(Bool, y[2], 2)
        @test pyeq(Bool, y[3], 3)
        @test_throws BoundsError y[-1]
        @test_throws BoundsError y[0]
        @test_throws BoundsError y[4]
        @test z[1] === 1
        @test z[2] === 2
        @test z[3] === 3
        @test_throws BoundsError z[-1]
        @test_throws BoundsError z[0]
        @test_throws BoundsError z[4]
    end
    @testset "copy" begin
        t = copy(z)
        @test t isa PyList{Int}
        push!(t, 99)
        @test z == [1, 2, 3]
        @test t == [1, 2, 3, 99]
    end
    @testset "setindex!" begin
        t = copy(z)
        @test setindex!(t, 11, 1) === t
        @test setindex!(t, 22.0, 2) === t
        @test setindex!(t, 66 // 2, 3) == t
        @test t == [11, 22, 33]
        @test_throws BoundsError t[-1] = 0
        @test_throws BoundsError t[0] = 0
        @test_throws BoundsError t[4] = 0
        @test t == [11, 22, 33]
        @test_throws Exception t[1] = nothing
        @test_throws Exception t[2] = missing
        @test_throws Exception t[3] = 4.5
        @test t == [11, 22, 33]
    end
    @testset "insert!" begin
        t = copy(z)
        @test insert!(t, 2, 11) === t
        @test t == [1, 11, 2, 3]
        @test insert!(t, 5, 33) === t
        @test t == [1, 11, 2, 3, 33]
        @test_throws BoundsError insert!(t, -1, 0)
        @test_throws BoundsError insert!(t, 0, 0)
        @test_throws BoundsError insert!(t, 7, 0)
        @test t == [1, 11, 2, 3, 33]
        @test_throws Exception insert!(t, nothing, 2)
        @test t == [1, 11, 2, 3, 33]
    end
    @testset "push!" begin
        t = copy(z)
        @test push!(t, 4) === t
        @test t == [1, 2, 3, 4]
        @test push!(t, 5, 6) === t
        @test t == [1, 2, 3, 4, 5, 6]
        @test_throws Exception push!(t, missing)
        @test t == [1, 2, 3, 4, 5, 6]
    end
    @testset "pushfirst!" begin
        t = copy(z)
        @test pushfirst!(t, -1) === t
        @test t == [-1, 1, 2, 3]
        @test pushfirst!(t, -3, -2) === t
        @test t == [-3, -2, -1, 1, 2, 3]
        @test_throws Exception pushfirst!(t, 4.5)
        @test t == [-3, -2, -1, 1, 2, 3]
    end
    @testset "append!" begin
        t = copy(z)
        @test append!(t, [4, 5, 6]) === t
        @test t == [1, 2, 3, 4, 5, 6]
        @test_throws Exception append!(t, [nothing, missing])
        @test t == [1, 2, 3, 4, 5, 6]
    end
    @testset "prepend!" begin
        t = copy(z)
        @test prepend!(t, [-3, -2, -1]) === t
        @test t == [-3, -2, -1, 1, 2, 3]
        @test_throws Exception append!(t, [nothing, missing])
        @test t == [-3, -2, -1, 1, 2, 3]
    end
    @testset "pop!" begin
        t = copy(z)
        @test pop!(t) == 3
        @test pop!(t) == 2
        @test pop!(t) == 1
        @test isempty(t)
        @test_throws BoundsError pop!(t)
    end
    @testset "popfirst!" begin
        t = copy(z)
        @test popfirst!(t) == 1
        @test popfirst!(t) == 2
        @test popfirst!(t) == 3
        @test isempty(t)
        @test_throws BoundsError popfirst!(t)
    end
    @testset "popat!" begin
        t = copy(z)
        @test_throws BoundsError popat!(t, 0)
        @test_throws BoundsError popat!(t, 4)
        @test t == [1, 2, 3]
        @test popat!(t, 2) == 2
        @test popat!(t, 2) == 3
        @test popat!(t, 1) == 1
        @test isempty(t)
        @test_throws BoundsError popat!(t, 1)
        @test_throws BoundsError popat!(t, 0)
        @test_throws BoundsError popat!(t, 1)
        @test_throws BoundsError popat!(t, 5)
    end
    @testset "reverse!" begin
        t = copy(z)
        @test reverse!(t) === t
        @test t == [3, 2, 1]
    end
    @testset "empty!" begin
        t = copy(z)
        @test !isempty(t)
        @test empty!(t) === t
        @test isempty(t)
    end
    @testset "construct empty" begin
        t = PyList{Int}()
        @test t isa PyList{Int}
        @test isempty(t)
        @test pyisinstance(t, pybuiltins.list)
    end
end

@testitem "PyPandasDataFrame" begin
    using Tables
    using DataFrames
    using CondaPkg
    CondaPkg.add("pandas")
    @test PyPandasDataFrame isa Type
    x = (x = [1, 2, 3], y = ["a", "b", "c"])
    py_df = pytable(x, :pandas)
    @test Tables.istable(PyTable(py_df))
    df = DataFrame(PyTable(py_df))
    @test df == DataFrame(x = [1, 2, 3], y = ["a", "b", "c"])

    x = PyPandasDataFrame(py_df)
    df = DataFrame(x)
    @test df == DataFrame(x = [1, 2, 3], y = ["a", "b", "c"])
    @test ispy(x)
    @test Tables.istable(x)
    @test Tables.columnaccess(x)
    @test Tables.columns(x)[:x] == [1, 2, 3]
    @test_throws Exception pyconvert(PyPandasDataFrame, 1)
    str = sprint(show, MIME("text/plain"), x)
    @test occursin(r"3×2 .*PyPandasDataFrame", str)
end

@testitem "PySet" begin
    x = pyset([1, 2, 3])
    y = PySet(x)
    z = PySet{Int}(x)
    e = PySet{String}()
    @testset "construct" begin
        @test y isa PySet{Py}
        @test z isa PySet{Int}
        @test e isa PySet{String}
        @test PythonCall.ispy(y)
        @test PythonCall.ispy(z)
        @test PythonCall.ispy(e)
        @test Py(y) === x
        @test Py(z) === x
        @test Py(e) !== x
    end
    @testset "length" begin
        @test length(y) == 3
        @test length(z) == 3
        @test length(e) == 0
    end
    @testset "isempty" begin
        @test !isempty(y)
        @test !isempty(z)
        @test isempty(e)
    end
    @testset "in" begin
        @test 1 in z
        @test 3 in z
        @test !(4 in z)
        @test !(1 in e)
        @test 2.0 in z
        @test !(2.1 in z)
    end
    @testset "push!" begin
        a = PySet{String}(["a"])
        @test a == Set(["a"])
        push!(a, "b")
        @test a == Set(["a", "b"])
        push!(a, "a")
        @test a == Set(["a", "b"])
    end
    @testset "delete!" begin
        a = PySet{Int}()
        @test a == Set()
        delete!(a, 0)
        @test a == Set()
        delete!(a, nothing)
        @test a == Set()
        push!(a, 1, 2, 3)
        @test a == Set([1, 2, 3])
        delete!(a, 2)
        @test a == Set([1, 3])
        delete!(a, 1.2)
        @test a == Set([1, 3])
        delete!(a, 3.0)
        @test a == Set([1])
        delete!(a, nothing)
        @test a == Set([1])
    end
    @testset "pop!" begin
        a = PySet{Int}()
        @test_throws Exception pop!(a)
        push!(a, 1, -1)
        x = pop!(a)
        @test x in [1, -1]
        @test a == Set([-x])
        y = pop!(a)
        @test x == -y
        @test a == Set()
        @test_throws Exception pop!(a)
        push!(a, 1, 2, 3)
        @test pop!(a, 1) == 1
        @test a == Set([2, 3])
        @test pop!(a, 2.0) == 2
        @test a == Set([3])
        @test_throws Exception pop!(a, 1)
        @test_throws Exception pop!(a, 1.0)
        @test_throws Exception pop!(a, nothing)
        @test a == Set([3])
        @test pop!(a, 1, 99) === 99
        @test pop!(a, 1.0, 99) === 99
        @test pop!(a, nothing, 99) === 99
    end
    @testset "empty!" begin
        a = PySet{Int}([1, 2, 3])
        @test a == Set([1, 2, 3])
        @test empty!(a) === a
        @test a == Set()
    end
    @testset "copy" begin
        z2 = copy(z)
        @test z2 isa PySet{Int}
        push!(z2, 99)
        @test z == Set([1, 2, 3])
        @test z2 == Set([1, 2, 3, 99])
    end
end

@testitem "PyTable" begin
    # TODO: figure out how to get pandas into the test environment
    @test PyTable isa Type
    @test_throws Exception PyTable(0)
end
