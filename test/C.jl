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

    @test !PythonCall.C.has_tstate()
    @test @pyregion pyconvert(Int, pyint(12)) == 12
    @test !PythonCall.C.has_tstate()

    # Nested regions reuse the state cached by the owning Julia thread. A break
    # must relinquish that cache while yielding, keep the task on the same Julia
    # thread, and restore both the cache and the task's original stickiness.
    test_task = current_task()
    oldsticky = test_task.sticky
    @pyregion begin
        local state = PythonCall.C.TASK_STATE()
        local thread_state = PythonCall.C.THREAD_STATE()
        local tid = threadid()
        @test test_task.sticky
        @test thread_state.task === test_task
        @test thread_state.task_state === state
        @pyregion @test PythonCall.C.current_task_state(test_task) === state
        @pyregionbreak begin
            @test test_task.sticky
            @test thread_state.task === nothing
            yield()
            @test threadid() == tid
        end
        @test thread_state.task === test_task
        @test thread_state.task_state === state
    end
    @test test_task.sticky == oldsticky
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

    # A task waiting in ordinary Julia code must release its Python state so that
    # another task on the same Julia thread can use Python. The bounded wait makes
    # a missing break fail promptly instead of leaving the test suite deadlocked.
    started = Channel{Nothing}(1)
    ready = Ref(false)
    break_status = Ref(:not_started)
    values = Ref((0, 0))
    @sync begin
        @async @pyregion begin
            put!(started, nothing)
            first = pyconvert(Int, pybuiltins.sum([1, 2, 3]))
            break_status[] = @pyregionbreak timedwait(() -> ready[], 2; pollint=0.001)
            second = pyconvert(Int, pybuiltins.sum([4, 5, 6]))
            values[] = (first, second)
        end
        @async begin
            take!(started)
            @test pyconvert(Int, pybuiltins.sum([7, 8, 9])) == 24
            ready[] = true
        end
    end
    @test break_status[] == :ok
    @test values[] == (6, 15)

    @test_throws ErrorException @pyregion @pyregionbreak error("region exception")
    @test !PythonCall.C.has_tstate()
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
