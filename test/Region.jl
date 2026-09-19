using TestItemRunner

@testitem "Python regions and task-safe thread states" setup = [Setup] begin
    using Base.Threads

    @test PythonCall.C.PyThreadState_GetUnchecked() == C_NULL
    @test @pyregion pyconvert(Int, pyint(12)) == 12
    @test PythonCall.C.PyThreadState_GetUnchecked() == C_NULL

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
