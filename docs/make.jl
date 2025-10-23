using Documenter, PythonCall

include("customdocs.jl")

makedocs(
    sitename = "PythonCall & JuliaCall",
    modules = [PythonCall],
    format = Documenter.HTML(assets = ["assets/favicon.ico"]),
    warnonly = [:missing_docs], # avoid raising error when docs are missing
    pages = [
        "Home" => "index.md",
        "The Julia module PythonCall" =>
            ["Guide" => "pythoncall.md", "Reference" => "pythoncall-reference.md"],
        "The Python module JuliaCall" =>
            ["Guide" => "juliacall.md", "Reference" => "juliacall-reference.md"],
        "Conversion" => [
            "Julia to Python" => "conversion-to-python.md",
            "Python to Julia" => "conversion-to-julia.md",
        ],
        "compat.md",
        "faq.md",
        "releasenotes.md",
        "v1-migration-guide.md",
    ],
)

deploydocs(repo = raw"github.com/JuliaPy/PythonCall.jl.git", push_preview = true)
