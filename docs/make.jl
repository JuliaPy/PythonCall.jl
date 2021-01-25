using Documenter, Python

makedocs(
    sitename = "Python.jl",
    modules = [Python],
)

deploydocs(
    repo = "github.com/cjdoris/Python.jl.git",
)
