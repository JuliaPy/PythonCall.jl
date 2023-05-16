using BenchmarkTools
using Printf
using PythonCall
using PythonCall: pydel!

# version-independent alias
# TODO: Is there anything inside PythonCall to access this executable's path?
# This is important to run the benchmark on the same interpreter that PythonCall
# feeds from
const PY_EXE_PATH = joinpath(ENV["CONDA_PREFIX"], "bin", "python")

@warn "Using " * read(`$(PY_EXE_PATH) --version`, String)

# Python imports
const py_random = pyimport("random")
const py_pyperf = pyimport("pyperf")

# Benchmark Suites
const SUITE = BenchmarkGroup()

include("suites/fill_dict.jl")

function main(suite)
    # Julia Benchmark
    tune!(suite)

    BenchmarkTools.save(
        joinpath(@__DIR__, "params.jl.json"),
        params(suite)
    )

    results = BenchmarkTools.run(suite; verbose=true)

    BenchmarkTools.save(
        joinpath(@__DIR__, "results.jl.json"),
        results
    )

    # Python Benchmark
    py_script_path  = joinpath(@__DIR__, "benchmark.py")
    py_results_path = joinpath(@__DIR__, "results.py.json")
    md_results_path = joinpath(@__DIR__, "RESULTS.md")

    # Remove results if exists
    rm(py_results_path; force=true)

    run(`$(PY_EXE_PATH) $(py_script_path) -o $(py_results_path)`)

    # Compare results and print output to markdown file
    py_results = py_pyperf.BenchmarkSuite.load(py_results_path)

    open(md_results_path, "w") do io

        println(io, raw"# PythonCall Benchmark")

        println(io)

        println(io, raw"## PythonCall vs. Python")

        println(io)

        # Start table
        print(io,
            raw"""
            | Task | Python | PythonCall | Multiplier |
            | :--- | :----: | :--------: | :--------: |
            """
        )

        for py_benchmark in py_results.get_benchmarks()
            name = py_benchmark.get_name()
            py_m = pyconvert(Float64, py_benchmark.median() * 10^3)
            jl_m = median(results[name]).time / 10^6
            ratio = jl_m / py_m

            @printf(io, "| `%s` | %.3fms | %.3fms | ×%4.2f |\n", name, py_m, jl_m, ratio)
        end

    end

    return nothing
end

main(SUITE)
