using TestItemRunner

@run_package_tests

@testmodule PyCall begin
    using PythonCall: PythonCall
    using Pkg: Pkg
    ENV["PYTHON"] = PythonCall.python_executable_path()
    @info "Building PyCall..." ENV["PYTHON"]
    Pkg.build("PyCall")
    using PyCall
end
