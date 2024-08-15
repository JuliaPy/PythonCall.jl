using TestItemRunner

# if you run tests in a conda environment, these env vars cause the aqua persistent tasks test to error
if haskey(ENV, "CONDA_PREFIX")
    delete!(ENV, "SSL_CERT_FILE")
    delete!(ENV, "SSL_CERT_DIR")
end

@run_package_tests
