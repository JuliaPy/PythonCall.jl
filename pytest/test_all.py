def test_import():
    import juliacall

def test_newmodule():
    import juliacall
    jl = juliacall.Main
    m = juliacall.newmodule("TestModule")
    assert isinstance(m, juliacall.Jl)
    assert jl.isa(m, jl.Module)
    assert str(jl.nameof(m)) == "TestModule"

def test_convert():
    import juliacall
    jl = juliacall.Main
    for (x, t) in [(None, jl.Nothing), (True, jl.Bool), ([1,2,3], jl.Vector)]:
        y = juliacall.convert(t, x)
        assert isinstance(y, juliacall.Jl)
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
