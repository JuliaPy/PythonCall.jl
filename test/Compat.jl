@testitem "gui" begin
    @testset "fix_qt_plugin_path" begin
        @test PythonCall.fix_qt_plugin_path() isa Bool
        # second time is a no-op
        @test PythonCall.fix_qt_plugin_path() === false
    end
    @testset "event_loop_on/off" begin
        for g in [:pyqt4, :pyqt5, :pyside, :pyside2, :pyside6, :gtk, :gtk3, :wx]
            # TODO: actually test the various GUIs somehow?
            @show g
            @test_throws PyException PythonCall.event_loop_on(g)
            @test PythonCall.event_loop_off(g) === nothing
        end
    end
end

@testitem "ipython" begin
    @testset "PythonDisplay" begin
        sys = pyimport("sys")
        io = pyimport("io")
        pystdout = sys.stdout
        fp = sys.stdout = io.StringIO()
        try
            d = PythonCall.Compat.PythonDisplay()
            @test display(d, 123) === nothing
            fp.seek(0)
            @test pyconvert(String, fp.read()) == "123\n"
        finally
            sys.stdout = pystdout
        end
    end
    @testset "IPythonDisplay" begin
        # TODO
    end
end

@testitem "multimedia" begin
    # TODO
end

@testitem "PyCall.jl" setup = [PyCall] begin
    x1 = pylist()
    x2 = PyCall.PyObject(x1)
    x3 = Py(x2)
    @test pyisinstance(x3, pybuiltins.list)
    @test pyis(x3, x1)
end

@testitem "Serialization.jl" begin
    using Serialization
    @testset "Py" begin
        for x in Py[
            Py(123),
            Py(1.23),
            Py("hello"),
            pylist([1, 2, 3]),
            pytuple([1, 2, 3]),
            Py(nothing),
            Py([1, 2, 3]),
            Py(:hello),
        ]
            io = IOBuffer()
            serialize(io, x)
            seekstart(io)
            y = deserialize(io)
            @test y isa Py
            @test pyis(pytype(x), pytype(y))
            @test pyeq(Bool, x, y)
        end
    end
    @testset "PyException" begin
        for e in Py[pybuiltins.ValueError("hello")]
            io = IOBuffer()
            x = PyException(e)
            serialize(io, x)
            seekstart(io)
            y = deserialize(io)
            @test y isa PyException
            @test pyeq(Bool, y.t, x.t)
            @test pyeq(Bool, y.v.args, x.v.args)
        end
    end
end

@testitem "Tables.jl" begin
    @testset "pytable" begin
        x = (x = [1, 2, 3], y = ["a", "b", "c"])
        # pandas
        # TODO: install pandas and test properly
        @test_throws PyException pytable(x, :pandas)
        # columns
        y = pytable(x, :columns)
        @test pyeq(Bool, y, pydict(x = [1, 2, 3], y = ["a", "b", "c"]))
        # rows
        y = pytable(x, :rows)
        @test pyeq(Bool, y, pylist([(1, "a"), (2, "b"), (3, "c")]))
        @test all(pyisinstance(y.x, pybuiltins.int) for y in y)
        @test all(pyisinstance(y.y, pybuiltins.str) for y in y)
        # rowdicts
        y = pytable(x, :rowdicts)
        @test pyeq(
            Bool,
            y,
            pylist([
                pydict(x = 1, y = "a"),
                pydict(x = 2, y = "b"),
                pydict(x = 3, y = "c"),
            ]),
        )
    end
end
