@testitem "bool → Bool" begin
    @test pyconvert(Bool, true) === true
    @test pyconvert(Bool, false) === false
    @test_throws Exception pyconvert(Bool, "hello")
end

@testitem "bool → Integer" begin
    @test pyconvert(Int, true) === 1
    @test pyconvert(Int, false) === 0
end

@testitem "bytes → Vector" begin
    x1 = pyconvert(Vector{UInt8}, pybytes(pylist([1, 2, 3])))
    @test x1 isa Vector{UInt8}
    @test x1 == [0x01, 0x02, 0x03]
    x2 = pyconvert(Base.CodeUnits, pybytes(pylist([102, 111, 111])))
    @test x2 isa Base.CodeUnits
    @test x2 == b"foo"
end

@testitem "complex → Complex" begin
    x1 = pyconvert(ComplexF32, pycomplex(1, 2))
    @test x1 === ComplexF32(1, 2)
    x2 = pyconvert(Complex, pycomplex(3, 4))
    @test x2 === ComplexF64(3, 4)
end

@testitem "float → Float" begin
    x1 = pyconvert(Float32, 12)
    @test x1 === Float32(12)
    x2 = pyconvert(Float64, 3.5)
    @test x2 === 3.5
end

@testitem "float → Nothing" begin
    @test_throws Exception pyconvert(Nothing, pyfloat(1.2))
    x1 = pyconvert(Nothing, pyfloat(NaN))
    @test x1 === nothing
end

@testitem "float → Missing" begin
    @test_throws Exception pyconvert(Missing, pyfloat(1.2))
    x1 = pyconvert(Missing, pyfloat(NaN))
    @test x1 === missing
end

@testitem "int → Integer" begin
    @test_throws Exception pyconvert(Int8, 300)
    @test_throws Exception pyconvert(UInt, -3)
    x1 = pyconvert(Int, 34)
    @test x1 === 34
    x2 = pyconvert(UInt8, 7)
    @test x2 === 0x07
    x3 = pyconvert(UInt32, typemax(UInt32))
    @test x3 === typemax(UInt32)
    x4 = pyconvert(Integer, big(3)^1000)
    @test x4 isa BigInt
    @test x4 == big(3)^1000
end

@testitem "None → Nothing" begin
    x1 = pyconvert(Nothing, pybuiltins.None)
    @test x1 === nothing
end

@testitem "None → Missing" begin
    x1 = pyconvert(Missing, pybuiltins.None)
    @test x1 === missing
end

@testitem "range → StepRange" begin
    x1 = pyconvert(StepRange, pyrange(10))
    @test x1 === (0:1:9)
    x2 = pyconvert(StepRange, pyrange(3, 9, 2))
    @test x2 === (3:2:7)
    x3 = pyconvert(StepRange, pyrange(20, 14, -1))
    @test x3 === (20:-1:15)
    x4 = pyconvert(StepRange, pyrange(30, -10, -3))
    @test x4 === (30:-3:-9)
end

@testitem "range → UnitRange" begin
    x1 = pyconvert(UnitRange, pyrange(10))
    @test x1 === (0:9)
    x2 = pyconvert(UnitRange, pyrange(3, 9, 1))
    @test x2 === (3:8)
end

@testitem "str → String" begin
    x1 = pyconvert(String, pystr("foo"))
    @test x1 === "foo"
    x2 = pyconvert(String, pystr("αβγℵ√"))
    @test x2 === "αβγℵ√"
end

@testitem "str → Symbol" begin
    x1 = pyconvert(Symbol, pystr("hello"))
    @test x1 === :hello
end

@testitem "str → Char" begin
    @test_throws Exception pyconvert(Char, pystr(""))
    @test_throws Exception pyconvert(Char, pystr("ab"))
    @test_throws Exception pyconvert(Char, pystr("abc"))
    x1 = pyconvert(Char, pystr("a"))
    @test x1 === 'a'
    x2 = pyconvert(Char, pystr("Ψ"))
    @test x2 === 'Ψ'
end

@testitem "iterable → Tuple" begin
    t1 = pyconvert(Tuple, (1, 2))
    @test t1 === (1, 2)
    t2 = pyconvert(Tuple{Vararg{Int}}, (3, 4, 5))
    @test t2 === (3, 4, 5)
    t3 = pyconvert(Tuple{Int,Int}, (6, 7))
    @test t3 === (6, 7)
    # generic case (>16 fields)
    t4 = pyconvert(Tuple{ntuple(i->Int,20)...,Vararg{Int}}, ntuple(i->i, 30))
    @test t4 === ntuple(i->i, 30)
end

@testitem "iterable → Vector" begin
    x1 = pyconvert(Vector, pylist([1, 2]))
    @test x1 isa Vector{Int}
    @test x1 == [1, 2]
    x2 = pyconvert(Vector, pylist([1, 2, nothing, 3]))
    @test x2 isa Vector{Union{Int,Nothing}}
    @test x2 == [1, 2, nothing, 3]
    x3 = pyconvert(Vector{Float64}, pylist([4, 5, 6]))
    @test x3 isa Vector{Float64}
    @test x3 == [4.0, 5.0, 6.0]
end

@testitem "iterable → Set" begin
    x1 = pyconvert(Set, pyset([1, 2]))
    @test x1 isa Set{Int}
    @test x1 == Set([1, 2])
    x2 = pyconvert(Set, pyset([1, 2, nothing, 3]))
    @test x2 isa Set{Union{Int,Nothing}}
    @test x2 == Set([1, 2, nothing, 3])
    x3 = pyconvert(Set{Float64}, pyset([4, 5, 6]))
    @test x3 isa Set{Float64}
    @test x3 == Set([4.0, 5.0, 6.0])
end

@testitem "iterable → Pair" begin
    @test_throws Exception pyconvert(Pair, ())
    @test_throws Exception pyconvert(Pair, (1,))
    @test_throws Exception pyconvert(Pair, (1, 2, 3))
    x1 = pyconvert(Pair, (2, 3))
    @test x1 === (2 => 3)
    x2 = pyconvert(Pair{String,Missing}, ("foo", nothing))
    @test x2 === ("foo" => missing)
end

@testitem "named tuple → NamedTuple" begin
    NT = pyimport("collections" => "namedtuple")
    t1 = pyconvert(NamedTuple, NT("NT", "x y")(1, 2))
    @test t1 === (x=1, y=2)
    @test_throws Exception pyconvert(NamedTuple, (2, 3))
    t2 = pyconvert(NamedTuple{(:x, :y)}, NT("NT", "x y")(3, 4))
    @test t2 === (x=3, y=4)
    @test_throws Exception pyconvert(NamedTuple{(:y, :x)}, NT("NT", "x y")(3, 4))
    t3 = pyconvert(NamedTuple{names,Tuple{Int,Int}} where {names}, NT("NT", "x y")(4, 5))
    @test t3 === (x=4, y=5)
    @test_throws Exception pyconvert(NamedTuple{names,Tuple{Int,Int}} where {names}, (5, 6))
    t4 = pyconvert(NamedTuple{(:x, :y),Tuple{Int,Int}}, NT("NT", "x y")(6, 7))
    @test t4 === (x=6, y=7)
end

@testitem "mapping → PyDict" begin
    x1 = pyconvert(PyDict, pydict([1=>11, 2=>22, 3=>33]))
    @test x1 isa PyDict{Any,Any}
    @test isequal(x1, Dict([1=>11, 2=>22, 3=>33]))
    x2 = pyconvert(PyDict{Int,Int}, pydict([1=>11, 2=>22, 3=>33]))
    @test x2 isa PyDict{Int,Int}
    @test x2 == Dict(1=>11, 2=>22, 3=>33)
end

@testitem "mapping → Dict" begin
    x1 = pyconvert(Dict, pydict(["a"=>1, "b"=>2]))
    @test x1 isa Dict{String, Int}
    @test x1 == Dict("a"=>1, "b"=>2)
    x2 = pyconvert(Dict{Char,Float32}, pydict(["c"=>3, "d"=>4]))
    @test x2 isa Dict{Char,Float32}
    @test x2 == Dict('c'=>3.0, 'd'=>4.0)
end

@testitem "sequence → PyList" begin
    x1 = pyconvert(PyList, pylist([1, 2, 3]))
    @test x1 isa PyList{Any}
    @test isequal(x1, [1, 2, 3])
    x2 = pyconvert(PyList{Int}, pylist([1, 2, 3]))
    @test x2 isa PyList{Int}
    @test x2 == [1, 2, 3]
end

@testitem "set → PySet" begin
    x1 = pyconvert(PySet, pyset([1, 2, 3]))
    @test x1 isa PySet{Any}
    @test isequal(x1, Set([1, 2, 3]))
    x2 = pyconvert(PySet{Int}, pyset([1, 2, 3]))
    @test x2 isa PySet{Int}
    @test x2 == Set([1, 2, 3])
end

@testitem "date → Date" begin
    using Dates
    x1 = pyconvert(Date, pydate(2001, 2, 3))
    @test x1 === Date(2001, 2, 3)
end

@testitem "time → Time" begin
    using Dates
    x1 = pyconvert(Time, pytime(12, 3, 4, 5))
    @test x1 === Time(12, 3, 4, 0, 5)
end

@testitem "datetime → DateTime" begin
    using Dates
    x1 = pyconvert(DateTime, pydatetime(2001, 2, 3, 4, 5, 6, 7000))
    @test x1 === DateTime(2001, 2, 3, 4, 5, 6, 7)
end

@testitem "pyconvert_add_rule (#364)" begin
    id = string(rand(UInt128), base=16)
    pyexec("""
    class Hello_364_$id:
        pass
    """, @__MODULE__)
    x = pyeval("Hello_364_$id()", @__MODULE__)
    @test pyconvert(Any, x) === x # This test has a side effect of influencing the rules cache
    t = pytype(x)
    PythonCall.pyconvert_add_rule("$(t.__module__):$(t.__qualname__)", String, (_, _) -> "Hello!!")
    @test pyconvert(String, x) == "Hello!!"
    @test pyconvert(Any, x) == "Hello!!" # Broken before PR #365
end

end
