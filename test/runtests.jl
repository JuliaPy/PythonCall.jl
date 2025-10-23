using TestItemRunner

# if you run tests in a conda environment, these env vars cause the aqua persistent tasks test to error
if haskey(ENV, "CONDA_PREFIX")
    delete!(ENV, "SSL_CERT_FILE")
    delete!(ENV, "SSL_CERT_DIR")
end

@testmodule Setup begin
    using PythonCall
    # test if we are in CI
    ci = get(ENV, "CI", "") == "true"
    # test if we have all dev conda deps
    devdeps = PythonCall.C.CTX.which == :CondaPkg
end

@run_package_tests
