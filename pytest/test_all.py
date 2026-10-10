import pytest


def test_import():
    import juliacall


def test_newmodule():
    import juliacall

    jl = juliacall.Main
    m = juliacall.newmodule("TestModule")
    assert isinstance(m, juliacall.Jl)
    assert jl.isa(m, jl.Module)
    assert str(jl.nameof(m)) == "TestModule"


def test_interactive():
    import juliacall

    juliacall.interactive(True)
    juliacall.interactive(False)


def test_JuliaError():
    import juliacall

    jl = juliacall.Main
    assert isinstance(juliacall.JuliaError, type)
    assert issubclass(juliacall.JuliaError, Exception)
    try:
        juliacall.Base.error("test error")
        err = None
    except juliacall.JuliaError as e:
        err = e
    assert err is not None
    assert isinstance(err, juliacall.JuliaError)
    exc = err.exception
    assert jl.isa(exc, jl.ErrorException)
    assert str(exc.msg) == "test error"
    bt = err.backtrace
    assert bt is not None


def test_issue_394():
    "https://github.com/JuliaPy/PythonCall.jl/issues/394"
    from juliacall import Main as jl

    x = 3
    f = lambda x: x + 1
    y = 5
    jl.x = x
    assert jl.x.jl_to_py() is x
    jl.f = f
    assert jl.f.jl_to_py() is f
    jl.y = y
    assert jl.y.jl_to_py() is y
    assert jl.x.jl_to_py() is x
    assert jl.f.jl_to_py() is f
    assert jl.y.jl_to_py() is y
    assert jl.jl_eval("f(x)").jl_to_py() == 4


def test_issue_433():
    "https://github.com/JuliaPy/PythonCall.jl/issues/433"
    from juliacall import Main as jl

    # Smoke test
    jl.jl_eval("x=1\nx=1")
    assert jl.x == 1

    # Do multiple things
    out = jl.jl_eval(
        """
        function _issue_433_g(x)
            return x^2
        end
        _issue_433_g(5)
        """
    )
    assert out == 25


def test_julia_gc():
    from juliacall import Main as jl

    if jl.jl_eval('v"1.11.0-" <= VERSION < v"1.11.3"'):
        # Seems to be a Julia bug - hopefully fixed in 1.11.3
        pytest.skip("Test not yet supported on Julia 1.11+")

    # We make a bunch of python objects with no reference to them,
    # then call GC to try to finalize them.
    # We want to make sure we don't segfault.
    # We also programmatically check things are working by verifying the queue is empty.
    # Debugging note: if you get segfaults, then run the tests with
    # `PYTHON_JULIACALL_HANDLE_SIGNALS=yes python3 -X faulthandler -m pytest -p no:faulthandler -s --nbval --cov=pysrc ./pytest/`
    # in order to recover a bit more information from the segfault.
    jl.jl_eval(
        """
        using PythonCall, Test
        let
            pyobjs = map(pylist, 1:100)
            Threads.@threads for obj in pyobjs
                finalize(obj)
            end
        end
        GC.gc()
        @test !isempty(PythonCall.GC.QUEUE.items)
        PythonCall.GC.gc()
        @test isempty(PythonCall.GC.QUEUE.items)
        """
    )


@pytest.mark.parametrize("yld", [True, False])
def test_parallel_call(yld):
    """Tests that ordinary calls execute Julia code in parallel."""
    from concurrent.futures import ThreadPoolExecutor, wait
    from time import time
    from juliacall import Main as jl

    # Julia implementations of sleep which do and do not yield.
    if yld:
        # use sleep, which yields
        jsleep = jl.sleep
    else:
        # use Libc.systemsleep which does not yield
        jsleep = jl.Libc.systemsleep
    assert not hasattr(jsleep, "jl_call_nogil")
    jyield = getattr(jl, "yield")
    # precompile
    jsleep(0.01)
    jyield()
    # use two threads
    pool = ThreadPoolExecutor(2)
    # run jsleep(1) twice concurrently
    t0 = time()
    fs = [pool.submit(jsleep, 1) for _ in range(2)]
    # submitting tasks should be very fast
    t1 = time() - t0
    assert t1 < 0.1
    # wait for the tasks to finish
    if yld:
        # we need to explicitly yield back to give the Julia scheduler a chance to
        # finish the sleep calls, so we yield every 0.1 seconds
        status = wait(fs, timeout=0.1)
        t2 = time() - t0
        while status.not_done:
            jyield()
            status = wait(fs, timeout=0.1)
            t2 = time() - t0
            assert t2 < 2.0
    else:
        wait(fs)
        t2 = time() - t0
    # executing the tasks should take about 1 second because they happen in parallel
    assert 0.9 < t2 < 1.5


def test_concurrent_callback_roundtrips():
    """Python threads can yield in Julia and repeatedly call back into Python."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, get_ident
    from juliacall import Main as jl

    # Each worker crosses Python -> Julia -> Python -> Julia repeatedly. Checking
    # the OS thread in the Python callback catches restoring another worker's
    # borrowed thread state, while the start barrier makes state races likely on
    # both GIL and free-threaded builds without assuming how Julia schedules them.
    jl.jl_eval(
        """
        function _callback_roundtrip(callback, python_tid, start)
            total = 0
            for value in start:(start + 19)
                yield()
                @assert !PythonCall.C.has_tstate()
                total += pyconvert(Int, callback(python_tid, value))
            end
            return total
        end
        """
    )

    def callback(expected_tid, value):
        assert get_ident() == expected_tid
        return 2 * int(jl.identity(value))

    # Compile every Julia/Python/Julia path before entering Julia concurrently from
    # foreign Python threads. This keeps the test focused on runtime state handoff.
    warmup_start = 100
    assert jl._callback_roundtrip(callback, get_ident(), warmup_start) == 2 * sum(
        range(warmup_start, warmup_start + 20)
    )

    participants = 2
    start_barrier = Barrier(participants)

    def worker(start):
        python_tid = get_ident()
        start_barrier.wait()
        return jl._callback_roundtrip(callback, python_tid, start)

    with ThreadPoolExecutor(participants) as pool:
        results = list(pool.map(worker, range(participants)))

    for start, total in enumerate(results):
        assert total == 2 * sum(range(start, start + 20))
