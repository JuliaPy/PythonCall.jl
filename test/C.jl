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
