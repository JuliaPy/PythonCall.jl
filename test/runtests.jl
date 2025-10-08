using TestItemRunner

@testmodule Setup begin
    using PythonCall
    # test if we are in CI
    ci = get(ENV, "CI", "") == "true"
    # test if we have all dev conda deps
    devdeps = PythonCall.C.CTX.which == :CondaPkg
end

@run_package_tests
