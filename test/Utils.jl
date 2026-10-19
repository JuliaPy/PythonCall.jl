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

@testitem "OncePerThread and OncePerTask" begin
    thread_count = Threads.Atomic{Int}(0)
    per_thread = PythonCall.Utils.OncePerThread{Int}() do
        Threads.atomic_add!(thread_count, 1)
        Threads.threadid()
    end
    @test per_thread() == Threads.threadid()
    @test per_thread() == Threads.threadid()
    @test thread_count[] == 1

    task_count = Threads.Atomic{Int}(0)
    per_task = PythonCall.Utils.OncePerTask{UInt}() do
        Threads.atomic_add!(task_count, 1)
        objectid(current_task())
    end
    @test per_task() == per_task()
    other_per_task = PythonCall.Utils.OncePerTask{UInt}(() -> typemax(UInt))
    @test other_per_task() == typemax(UInt)
    other = fetch(Threads.@spawn (per_task(), per_task()))
    @test other[1] == other[2]
    @test task_count[] == 2
end
