@testitem "libstdc++ version" begin
    ENV["JULIA_CONDAPKG_LIBSTDCXX_VERSION_BOUND"] = ">=3.4,<=12"

    cxxversion = PythonCall.C.get_libstdcxx_version_bound()
    @test cxxversion == ">=3.4,<=12"
end