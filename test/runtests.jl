using TestItemRunner

@testmodule Setup begin
    # test if we are in CI
    ci = get(ENV, "CI", "") == "true"
    # test if we have all dev conda deps (assume true if not in CI)
    devdeps = ci ? ENV["JULIA_PYTHONCALL_EXE"] == "@CondaPkg" : true
end

@run_package_tests
