@testitem "python info" begin
    @testset "python_executable_path" begin
        @test PythonCall.python_executable_path() isa String
        @test occursin("python", PythonCall.python_executable_path())
    end
    @testset "python_library_path" begin
        @test PythonCall.python_library_path() isa String
        @test occursin("python", PythonCall.python_library_path())
    end
    @testset "python_library_handle" begin
        @test PythonCall.python_library_handle() isa Ptr{Cvoid}
        @test PythonCall.python_library_handle() != C_NULL
    end
    @testset "python_version" begin
        @test PythonCall.python_version() isa VersionNumber
        @test PythonCall.python_version().major == 3
    end
end

@testitem "Python regions and task-safe thread states" setup = [Setup] begin
    using Base.Threads

    @test PythonCall.C.PyThreadState_GetUnchecked() == C_NULL
    @test @pyregion pyconvert(Int, pyint(12)) == 12
    @test PythonCall.C.PyThreadState_GetUnchecked() == C_NULL

    # Nested regions reuse the state cached by the owning Julia thread. A break
    # must relinquish that cache while yielding and restore it on return.
    @pyregion begin
        local state = PythonCall.C.TASK_STATE()
        local thread_state = PythonCall.C.THREAD_STATE()
        @test thread_state.task === current_task()
        @test thread_state.task_state === state
        @pyregion @test PythonCall.C.current_task_state(current_task()) === state
        @pyregionbreak begin
            @test thread_state.task === nothing
            yield()
        end
        @test thread_state.task === current_task()
        @test thread_state.task_state === state
    end
    @test PythonCall.C.THREAD_STATE().task === nothing

    @test @pyregion begin
        @pyregion pyconvert(Int, pyint(1)) == 1
        @pyregionbreak begin
            yield()
            @pyregion pyconvert(Int, pyint(2)) == 2
            @pyregionbreak yield()
        end
        pyconvert(Int, pyint(3)) == 3
    end

    @test_throws ErrorException @pyregion @pyregionbreak error("region exception")
    @test PythonCall.C.PyThreadState_GetUnchecked() == C_NULL
    @test_throws PyException @pyregion pybuiltins.int("not an integer")
    @test pyconvert(Int, pyint(5)) == 5

    results = fetch.([@spawn begin
        total = 0
        for i in 1:100
            total += pyconvert(Int, pyint(i))
            @pyregionbreak yield()
        end
        total
    end for _ in 1:max(8, 2nthreads())])
    @test all(==(5050), results)

    f = pyfunc() do
        yield()
        pyconvert(Int, pybuiltins.sum([1, 2, 3]))
    end
    @test pyconvert(Int, f()) == 6

    @test pyconvert(Int, @py 1 + @jl(pyconvert(Int, pyint(2)))) == 3
end
