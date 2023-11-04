@testitem "pywith" begin
    @testset "no error" begin
        tdir = pyimport("tempfile").TemporaryDirectory()
        tname = pyconvert(String, tdir.name)
        @test isdir(tname)
        pywith(tdir) do name
            @test pyconvert(String, name) == tname
        end
        @test !isdir(tname)
    end
    @testset "error" begin
        tdir = pyimport("tempfile").TemporaryDirectory()
        tname = pyconvert(String, tdir.name)
        @test isdir(tname)
        @test_throws PyException pywith(name -> name.invalid_attr, tdir)
        @test !isdir(tname)
    end
end

@testitem "gui" begin
    @test PythonCall.fix_qt_plugin_path() isa Bool
    @test PythonCall.fix_qt_plugin_path() === false
    for g in [:pyqt4, :pyqt5, :pyside, :pyside2, :gtk, :gtk3, :wx, :tkinter]
        # TODO: actually test the various GUIs somehow?
        @test_throws PyException PythonCall.event_loop_on(g)
        @test PythonCall.event_loop_off(g) === nothing
    end
end
