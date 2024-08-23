@testitem "mimes_for" begin
    # this example from https://github.com/JuliaPy/PythonCall.jl/issues/487
    struct Test{T<:Number}
        x::T
    end
    Base.show(io::IO, ::MIME"text/plain", x::Test{T}) where {T} = show(io, x.t)
    Base.show(io::IO, ::MIME"text/x-test", x::Test) = show(io, x.t)

    @testset for x in Any[1, "foo", [], 'z', Test(5)]
        mimes = PythonCall.Utils.mimes_for(x)
        @test mimes isa Vector{String}
        @test "text/plain" in mimes
        @test "text/html" in mimes
        @test ("text/x-test" in mimes) == (x isa Test)
    end
end

@testitem "StaticString length and indexing" begin
    s = PythonCall.Utils.StaticString{UInt32,44}("ababababb")
    @test length(s) == 9
    @test s[1] == 'a'
    @test s[1:2] == "ab"
    @test s[1:2:end] == "aaaab"
end
