using Documenter, Python

makedocs(
    sitename = "Python.jl",
    modules = [Python],
    pages = [
        "Home" => "index.md",
        "getting-started.md",
        "pythonjl.md",
        "juliapy.md",
        "conversion.md",
        "compat.md",
    ]
)

deploydocs(
    repo = "github.com/cjdoris/Python.jl.git",
)
