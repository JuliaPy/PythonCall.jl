using Documenter, PythonCall, Markdown

include("customdocs.jl")

makedocs(
    sitename = "PythonCall & JuliaCall",
    modules = [PythonCall],
    warnonly = [:missing_docs], # avoid raising error when docs are missing
    pages = [
        "Home" => "index.md",
        "The Julia module PythonCall" => [
            "Guide" => "pythoncall.md",
            "Reference" => "pythoncall-reference.md",
        ],
        "The Python module JuliaCall" => [
            "Guide" => "juliacall.md",
            "Reference" => "juliacall-reference.md",
        ],
        "Conversion" => [
            "Julia to Python" => "conversion-to-python.md",
            "Python to Julia" => "conversion-to-julia.md",
        ],
        "compat.md",
        "faq.md",
        "pycall.md",
        "releasenotes.md",
    ]
)

deploydocs(
    repo = "github.com/JuliaPy/PythonCall.jl.git",
)
