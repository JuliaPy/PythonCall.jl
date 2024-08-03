def test_import():
    import juliacall

def test_newmodule():
    import juliacall
    jl = juliacall.Main
    m = juliacall.newmodule("TestModule")
    assert isinstance(m, juliacall.ModuleValue)
    assert jl.isa(m, jl.Module)
    assert str(jl.nameof(m)) == "TestModule"

def test_convert():
    import juliacall
    jl = juliacall.Main
    for (x, t) in [(None, jl.Nothing), (True, jl.Bool), ([1,2,3], jl.Vector)]:
        y = juliacall.convert(t, x)
        assert isinstance(y, juliacall.AnyValue)
        assert jl.isa(y, t)

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
    f = lambda x: x+1
    y = 5
    jl.x = x
    assert jl.x is x
    jl.f = f
    assert jl.f is f
    jl.y = y
    assert jl.y is y
    assert jl.x is x
    assert jl.f is f
    assert jl.y is y
    assert jl.seval("f(x)") == 4

def test_issue_433():
    "https://github.com/JuliaPy/PythonCall.jl/issues/433"
    from juliacall import Main as jl

    # Smoke test
    jl.seval("x=1\nx=1")
    assert jl.x == 1

    # Do multiple things
    out = jl.seval(
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
    # We make a bunch of python objects with no reference to them,
    # then call GC to try to finalize them.
    # We want to make sure we don't segfault.
    # We also programmatically check things are working by verifying the queue is empty.
    # Debugging note: if you get segfaults, then run the tests with
    # `PYTHON_JULIACALL_HANDLE_SIGNALS=yes python3 -X faulthandler -m pytest -p no:faulthandler -s --nbval --cov=pysrc ./pytest/`
    # in order to recover a bit more information from the segfault.
    jl.seval(
        """
        using PythonCall, Test
        let
            pyobjs = map(pylist, 1:100)
            Threads.@threads for obj in pyobjs
                finalize(obj)
            end
        end
        GC.gc()
        @test isempty(PythonCall.GC.QUEUE.items)
        """
    )

def test_call_nogil():
    """Tests that we can execute Julia code in parallel by releasing the GIL."""
    from concurrent.futures import ThreadPoolExecutor, wait
    from time import time
    from juliacall import Main as jl
    # julia implementation of sleep which releases the GIL
    # this test uses Base.Libc.systemsleep which does not yield to the scheduler
    jsleep = jl.Libc.systemsleep._jl_call_nogil
    # precompile
    jsleep(0.01)
    # use two threads
    pool = ThreadPoolExecutor(2)
    # run sleep twice concurrently
    t0 = time()
    fs = [pool.submit(jsleep, 1) for _ in range(2)]
    t1 = time() - t0
    wait(fs)
    t2 = time() - t0
    # submitting tasks should be very fast
    assert t1 < 0.1
    # executing the tasks should take about 1 second because they happen in parallel
    assert 0.9 < t2 < 1.5

def test_call_nogil_yielding():
    """Same as the previous test but with a function (sleep) that yields.
    
    Yielding puts us back into Python, which itself doesn't ever yield back to Julia, so
    the function can never return. Hence for the threads to finish, we need to
    explicitly yield back to Julia.
    """
    from concurrent.futures import ThreadPoolExecutor, wait
    from time import sleep, time
    from juliacall import Main as jl
    # julia implementation of sleep which releases the GIL
    # in this test we use Base.sleep which yields to the scheduler
    jsleep = jl.sleep._jl_call_nogil
    jyield = getattr(jl, "yield")
    # precompile
    jsleep(0.01)
    jyield()
    # use two threads
    pool = ThreadPoolExecutor(2)
    # run sleep twice concurrently
    t0 = time()
    fs = [pool.submit(jsleep, 1) for _ in range(2)]
    t1 = time() - t0
    # because sleep() yields to the scheduler, which puts us back in Python, we need to
    # explicitly yield back to give the scheduler a chance to finish the sleep calls, so
    # we yield every 0.1 seconds
    status = wait(fs, timeout=0.1)
    t2 = time() - t0
    while t2 < 2.0 and status.not_done:
        jyield()
        status = wait(fs, timeout=0.1)
        t2 = time() - t0
    # submitting tasks should be very fast
    assert t1 < 0.1
    # the tasks should have finished
    assert not status.not_done
    # executing the tasks should take about 1 second because they happen in parallel
    assert 0.9 < t2 < 1.5
