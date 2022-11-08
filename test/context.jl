@testitem "libstdc++ version" begin
    cxxversion = PythonCall.get_libstdcxx_version_bound()

    if VERSION <= v"1.6.2"
        @test cxxversion == ">=3.4,<9.2"
    else
        @test cxxversion == ">=3.4,<11.4"
    end

    ENV["JULIA_CONDAPKG_LIBSTDCXX_VERSION_BOUND"] = ">=3.4,<=12"

    cxxversion = PythonCall.get_libstdcxx_version_bound()
    @test cxxversion == ">3.4,<=12"
end