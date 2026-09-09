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

@testitem "exe = @ENV" begin
    mktempdir() do dir
        # A throwaway environment prepended to the load path overrides the `exe`
        # preference for a child process while still resolving PythonCall from ours.
        write(
            joinpath(dir, "Project.toml"),
            "[extras]\nPythonCall = \"6099a3de-0909-46bc-b1f4-468b9a2dfc0d\"\n",
        )
        write(joinpath(dir, "LocalPreferences.toml"), "[PythonCall]\nexe = \"@ENV\"\n")
        sep = Sys.iswindows() ? ";" : ":"
        loadpath = join([dir, Base.active_project(), "@stdlib"], sep)
        exe = PythonCall.python_executable_path()
        code = "using PythonCall; print(PythonCall.python_executable_path())"
        cmd = `$(Base.julia_cmd()) --startup-file=no -e $code`

        # variable set: it is used as the executable
        withset = addenv(cmd, "JULIA_LOAD_PATH" => loadpath, "JULIA_PYTHONCALL_EXE" => exe)
        @test readchomp(withset) == exe

        # variable unset: fails at load time with a clear message
        unset = addenv(cmd, "JULIA_LOAD_PATH" => loadpath, "JULIA_PYTHONCALL_EXE" => nothing)
        err = IOBuffer()
        proc = run(pipeline(ignorestatus(unset), stdout = devnull, stderr = err))
        @test !success(proc)
        @test occursin("JULIA_PYTHONCALL_EXE is not set", String(take!(err)))
    end
end
