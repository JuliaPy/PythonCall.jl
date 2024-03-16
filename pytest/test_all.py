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
